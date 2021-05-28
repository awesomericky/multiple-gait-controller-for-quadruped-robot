//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <cstdlib>
#include <cmath>
#include <set>
#include "../../RaisimGymEnv.hpp"

// To make new function
// 1. Environment.hpp
// 2. raisim_gym.cpp
// 3. RaisimGymEnv.hpp
// 4. VectorizedEnvironment.hpp
// 5. RaisimGymVecEnv.py (if needed)


namespace raisim
{

    class ENVIRONMENT : public RaisimGymEnv
    {

    public:
        explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) : RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable)
        {

            /// add objects
            anymal_ = world_->addArticulatedSystem(resourceDir_ + "/laikago/laikago.urdf");
            anymal_->setName("anymal");
            anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            world_->addGround();

            /// get robot data
            gcDim_ = anymal_->getGeneralizedCoordinateDim(); // 15 (w/ fixed HAA joints)
            gvDim_ = anymal_->getDOF();                      // 14 (w/ fixed HAA joints)
            nJoints_ = gvDim_ - 6;                           // 8 (w/ fixed HAA joints)

            /// initialize containers
            gc_.setZero(gcDim_);
            gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_);
            gv_init_.setZero(gvDim_);
            pTarget_.setZero(gcDim_);
            vTarget_.setZero(gvDim_);
            pTarget12_.setZero(nJoints_);

            leg_work.setZero(4);

            /// desired velocity
            desired_velocity = 0.0;

            /// reward constant
            reward_torque_coeff = cfg["reward"]["torque"]["coeff"].As<double>();
            reward_velocity_coeff = cfg["reward"]["forwardVel_difference"]["coeff"].As<double>();
            reward_height_coeff = cfg["reward"]["height"]["coeff"].As<double>();
            reward_orientation_coeff = cfg["reward"]["orientation"]["coeff"].As<double>();
            // reward_impulse_coeff = cfg["reward"]["impulse"]["coeff"].As<double>();
            // reward_leg_work_coeff = cfg["reward"]["leg_work_entropy"]["coeff"].As<double>();
            CPG_reward_velocity_coeff = cfg["CPG_reward"]["forwardVel_difference"]["coeff"].As<double>();
            CPG_reward_GRF_coeff = cfg["CPG_reward"]["GRF_entropy"]["coeff"].As<double>();

            height_threshold = cfg["reward"]["height"]["coeff"].As<double>();

            /// contact foot index
            contact_foot_idx.insert(anymal_->getBodyIdx("FR_calf"));
            contact_foot_idx.insert(anymal_->getBodyIdx("FL_calf"));
            contact_foot_idx.insert(anymal_->getBodyIdx("RR_calf"));
            contact_foot_idx.insert(anymal_->getBodyIdx("RL_calf"));

            // nominal configuration of laikago
            gc_init_ << 0, 0, 0.46, 1, 0.0, 0.0, 0.0, 0.5, -1, 0.5, -1, 0.5, -1, 0.5, -1;

            /// set pd gains
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero();
            jointPgain.tail(nJoints_).setConstant(40.0); //20.0
            jointDgain.setZero();
            jointDgain.tail(nJoints_).setConstant(1.0); //0.2
            anymal_->setPdGains(jointPgain, jointDgain);
            anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 26;
            actionDim_ = nJoints_;
            actionMean_.setZero(actionDim_);
            actionStd_.setZero(actionDim_);
            obDouble_.setZero(obDim_);

            /// action scaling
            actionMean_ = gc_init_.tail(nJoints_);
            actionStd_.setConstant(0.6);

            /// Reward coefficients
            rewards_.initializeFromConfigurationFile(cfg["reward"]);

            /// indices of links that should not make contact with ground
            footIndices_.insert(anymal_->getBodyIdx("FR_calf")); // 2
            footIndices_.insert(anymal_->getBodyIdx("FL_calf")); // 4
            footIndices_.insert(anymal_->getBodyIdx("RR_calf")); // 6
            footIndices_.insert(anymal_->getBodyIdx("RL_calf")); // 8

            /// visualize if it is the first environment
            if (visualizable_)
            {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();
                server_->focusOn(anymal_);
            }

        }

        void init() final {}

        void reset() final
        {
            anymal_->setState(gc_init_, gv_init_);
            updateObservation();
        }

        float step(const Eigen::Ref<EigenVec> &action) final
        {
            /// action scaling
            pTarget12_ = action.cast<double>();
            pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
            pTarget12_ += actionMean_;
            pTarget_.tail(nJoints_) = pTarget12_;

            anymal_->setPdTarget(pTarget_, vTarget_);

            for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++)
            {
                if (server_)
                    server_->lockVisualizationServerMutex();
                world_->integrate();
                if (server_)
                    server_->unlockVisualizationServerMutex();
            }

            updateObservation();

            torque = anymal_->getGeneralizedForce().e(); // squaredNorm

            // Real
            joint_work = (gv_.tail(8) * torque.tail(8)).array().abs() * control_dt_;
            // leg_work << joint_work[0] + joint_work[1], joint_work[2] + joint_work[3], joint_work[4] + joint_work[5], joint_work[6] + joint_work[7];
            // leg_work /= leg_work.sum();
            // leg_work += 1e-6;
            // leg_work_entropy = -(leg_work * leg_work.log()).sum();
            rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
            rewards_.record("forwardVel_difference", std::exp(-std::abs(bodyLinearVel_[0] - desired_velocity)));

            if (gc_[2] > gc_init_[2] + height_threshold) {
                height_violation = gc_[2] - (gc_init_[2] + height_threshold);
            }
            else if (gc_[2] < gc_init_[2] - height_threshold) {
                height_violation = (gc_init_[2] - height_threshold) - gc_[2];
            }
            else {
                height_violation = 0;
            }
            rewards_.record("height", std::exp(-height_violation));
            rewards_.record("orientation", std::exp(-std::abs(pitch_and_yaw - 1)));
            // rewards_.record("leg_work_entropy", leg_work_entropy);
            // rewards_.record("uncontact_penalty", unContactPenalty);

            CPG_rewards_ = GRF_entropy * CPG_reward_GRF_coeff + std::exp(-std::abs(bodyLinearVel_[0] - desired_velocity)) * CPG_reward_velocity_coeff;
            // CPG_rewards_.record("GRF_entropy", GRF_entropy);
            // CPG_rewards_.record("forwardVel_difference", std::exp(-std::abs(bodyLinearVel_[0] - desired_velocity)));

            return rewards_.sum();
        }

        void reward_logging(Eigen::Ref<EigenVec> rewards) final
        {
            reward_log.setZero(5);  ///////// Need to change!! Don't forget!! /////////////
            reward_log[0] = anymal_->getGeneralizedForce().squaredNorm() * reward_torque_coeff;
            reward_log[1] = std::exp(-std::abs(bodyLinearVel_[0] - desired_velocity)) * reward_velocity_coeff;
            reward_log[2] = std::exp(-height_violation) * reward_height_coeff;
            reward_log[3] = std::exp(-std::abs(pitch_and_yaw - 1)) * reward_orientation_coeff;
            // reward_log[4] = leg_work_entropy * reward_leg_work_coeff;
            // reward_log[5] = unContactPenalty;
            reward_log[4] = GRF_entropy * CPG_reward_GRF_coeff;
            // reward_log[5] = std::exp(-std::abs(bodyLinearVel_[0] - desired_velocity)) * CPG_reward_velocity_coeff;
            // reward_log[7] = GRF_impulse_reward * reward_impulse_coeff;

            rewards = reward_log.cast<float>();
        }

        void contact_logging(Eigen::Ref<EigenVec> contacts) final
        {
            contacts = GRF_impulse.cast<float>();
        }

        void set_target_velocity(Eigen::Ref<EigenVec> velocity) final
        {
            desired_velocity = velocity[0];
        }

        void get_CPG_reward(Eigen::Ref<EigenVec> CPG_reward) final
        {
            CPG_reward[0] = CPG_rewards_;
        }

        void updateObservation()
        {
            anymal_->getState(gc_, gv_);

            raisim::Vec<4> quat;
            raisim::Mat<3, 3> rot;
            quat[0] = gc_[3];
            quat[1] = gc_[4];
            quat[2] = gc_[5];
            quat[3] = gc_[6];
            raisim::quatToRotMat(quat, rot);
            bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

            obDouble_ << gc_[2],                 /// body height // dim=1
                rot.e().row(2).transpose(),      /// body orientation // dim=3
                gc_.tail(8),                     /// joint angles // dim=8 (w/ HAA joint fixed)
                bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity // dim=6 (3 + 3)
                gv_.tail(8);                     /// joint velocity // dim=8 (w/ HAA joint fixed)

            pitch_and_yaw = rot.e().row(2).transpose()[2];

            /// z axis contact impulse for each feet (= perpendicular GRF * dt)
            total_contact_impulse.setZero(4);  // only perpendicular GRF
            GRF_impulse.setZero(4);  // total GRF

            // unContactPenalty = -5.;

            // check all contacts
            for (auto &contact : anymal_->getContacts())
            {
                if (contact.skip())
                    continue; /// if the contact is internal, one contact point is set to 'skip'

                single_contact_impulse = contact.getContactFrame().e().transpose() * contact.getImpulse()->e();

                if (contact_foot_idx.find(contact.getlocalBodyIndex()) != contact_foot_idx.end())
                {
                    idx = int(int(contact.getlocalBodyIndex()) / 2) - 1;
                    total_contact_impulse[idx] = std::max(double(single_contact_impulse[2]), 0.0);
                    GRF_impulse[idx] = single_contact_impulse.squaredNorm() * control_dt_;
                    // unContactPenalty = 0.;
                }
            }

            if (total_contact_impulse.sum() < 1e-4) {
                /// almost no contact between foot and ground
                GRF_entropy = 0.0;
            }
            else {
                /// compute perpendicular GRF entropy
                total_contact_impulse = total_contact_impulse / total_contact_impulse.sum();
                total_contact_impulse = total_contact_impulse + 1e-6;
                GRF_entropy = -(total_contact_impulse * total_contact_impulse.log()).sum();
            }

            // if (GRF_impulse.sum() < 1e-4) {
            //     /// almost no contact between foot and ground
            //     GRF_impulse_reward = 0.0;
            // }
            // else {
            //     GRF_impulse_reward = 1 / (GRF_impulse.sum() / 4);
            // }


        }

        void observe(Eigen::Ref<EigenVec> ob) final
        {
            /// convert it to float
            ob = obDouble_.cast<float>();
        }

        bool isTerminalState(float &terminalReward) final
        {
            terminalReward = float(terminalRewardCoeff_);

            /// if the contact body is not feet
            for (auto &contact : anymal_->getContacts())
                if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
                    return true;

            terminalReward = 0.f;
            return false;
        }

    private:
        int gcDim_, gvDim_, nJoints_, idx;
        bool visualizable_ = false;
        raisim::ArticulatedSystem *anymal_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque;
        double terminalRewardCoeff_ = -10., velocity, desired_velocity, reward_torque_coeff, pitch_and_yaw, leg_work_entropy;
        double reward_velocity_coeff, reward_impulse_coeff, reward_height_coeff, reward_orientation_coeff, GRF_entropy, GRF_impulse_reward, reward_leg_work_coeff;
        double CPG_reward_GRF_coeff, CPG_reward_velocity_coeff, CPG_rewards_;
        double unContactPenalty = -5., height_threshold, height_violation;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_, reward_log;
        Eigen::VectorXd single_contact_impulse;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
        Eigen::ArrayXd total_contact_impulse, GRF_impulse, joint_work, leg_work;
        std::set<size_t> footIndices_, contact_foot_idx;
    };
}
