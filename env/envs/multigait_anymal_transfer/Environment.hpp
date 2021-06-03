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
// 2. raisim_gym.cpp (if needed)
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
            // next_gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_);
            gv_init_.setZero(gvDim_);
            pTarget_.setZero(gcDim_);
            vTarget_.setZero(gvDim_);
            pTarget12_.setZero(nJoints_);
            previous_action.setZero(nJoints_);
            current_action.setZero(nJoints_);

            footPos_W.resize(4);
            footVel_W.resize(4);
            footContactVel_.resize(4);
            // footNormal_.resize(4);

            /// desired velocity
            desired_velocity = 0.0;

            /// reward constant
            // reward_torque_coeff = cfg["reward"]["torque"]["coeff"].As<double>();
            // reward_velocity_coeff = cfg["reward"]["forwardVel_difference"]["coeff"].As<double>();
            reward_height_coeff = cfg["reward"]["height"]["coeff"].As<double>();
            // reward_orientation_coeff = cfg["reward"]["orientation"]["coeff"].As<double>();
            // reward_impulse_coeff = cfg["reward"]["impulse"]["coeff"].As<double>();
            // reward_leg_work_coeff = cfg["reward"]["leg_work_entropy"]["coeff"].As<double>();
            // CPG_reward_velocity_coeff = cfg["CPG_reward"]["forwardVel_difference"]["coeff"].As<double>();
            // CPG_reward_GRF_coeff = cfg["CPG_reward"]["GRF_entropy"]["coeff"].As<double>();
            // CPG_reward_torque_coeff = cfg["CPG_reward"]["torque"]["coeff"].As<double>();

            height_threshold = cfg["reward"]["height"]["threshold"].As<double>();

            /// contact foot index
            contact_foot_idx.insert(anymal_->getBodyIdx("FR_calf"));
            contact_foot_idx.insert(anymal_->getBodyIdx("FL_calf"));
            contact_foot_idx.insert(anymal_->getBodyIdx("RR_calf"));
            contact_foot_idx.insert(anymal_->getBodyIdx("RL_calf"));

            foot_idx[0] = anymal_->getFrameIdxByName("FR_foot_fixed");  // 14
            foot_idx[1] = anymal_->getFrameIdxByName("FL_foot_fixed");  // 16
            foot_idx[2] = anymal_->getFrameIdxByName("RR_foot_fixed");  // 18
            foot_idx[3] = anymal_->getFrameIdxByName("RL_foot_fixed");  // 20

            // nominal configuration of laikago
            gc_init_ << 0, 0, 0.46, 1, 0.0, 0.0, 0.0, 0.5, -1, 0.5, -1, 0.5, -1, 0.5, -1;
            // update_initial_state(gc_init_);

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

        void reset_w_previous()
        {
            anymal_->setState(next_gc_init_, gv_init_);

            current_n_state = 0;

            updateObservation();
        }

        float step(const Eigen::Ref<EigenVec> &action) final
        {
            /// action scaling
            pTarget12_ = action.cast<double>();
            current_action = action.cast<double>();
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

            calculate_cost();

            if (gc_[2] > gc_init_[2] + height_threshold) {
                height_violation = gc_[2] - (gc_init_[2] + height_threshold);
            }
            else if (gc_[2] < gc_init_[2] - height_threshold) {
                height_violation = (gc_init_[2] - height_threshold) - gc_[2];
            }
            else {
                height_violation = 0;
            }

            rewards_.record("joint_torque", -torqueCost);
            rewards_.record("linear_vel_error", -linvelCost);
            rewards_.record("angular_vel_error", -angVelCost);
            rewards_.record("foot_clearance", -footClearanceCost);
            rewards_.record("foot_slip", -slipCost);
            rewards_.record("foot_z_vel", -footVelCost);
            rewards_.record("joint_vel", -velLimitCost);
            rewards_.record("previous_action_smooth", -previousActionCost);
            rewards_.record("orientation", -orientationCost);
            rewards_.record("leg_phase", -leg_phase_cost);
            rewards_.record("height", std::exp(height_violation));

            CPG_rewards_ = -cost;

            previous_action = action.cast<double>();

            return rewards_.sum();
        }

        void reward_logging(Eigen::Ref<EigenVec> rewards) final
        {
            reward_log.setZero(12);  ///////// Need to change!! Don't forget!! /////////////
            reward_log[0] = -torqueCost;
            reward_log[1] = -linvelCost;
            reward_log[2] = -angVelCost;
            reward_log[3] = -footClearanceCost;
            reward_log[4] = -slipCost;
            reward_log[5] = -footVelCost;
            reward_log[6] = -velLimitCost;
            reward_log[7] = -previousActionCost;
            reward_log[8] = -orientationCost;
            reward_log[9] = -leg_phase_cost;
            reward_log[10] = std::exp(-height_violation);
            reward_log[11] = costScale_;

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

        void get_CPG_reward(Eigen::Ref<EigenVec> CPG_reward) 
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

            pitch_and_yaw = rot.e().row(2).transpose();
            GRF_impulse.setZero(4);

            comprehend_contacts();

            // if (current_n_state == next_initialize_n_state)
            //     update_initial_state(gc_);
            // current_n_state += 1;

        }

        void observe(Eigen::Ref<EigenVec> ob) final
        {
            /// convert it to float
            ob = obDouble_.cast<float>();
        }

        void comprehend_contacts()
        {
            numContact_ = anymal_->getContacts().size();

            numFootContact_ = 0;
            numBodyContact_ = 0;
            numBaseContact_ = 0;

            for (int k = 0; k < 4; k++)
            {
                footContactState_[k] = false;
                anymal_->getFramePosition(foot_idx[k], footPos_W[k]);  //position of the feet
                anymal_->getFrameVelocity(foot_idx[k], footVel_W[k]);
            }

            raisim::Vec<3> vec3;

            //Classify foot contact
            if (numContact_ > 0)
            {
                for (int k = 0; k < numContact_; k++)
                {
                    if (!anymal_->getContacts()[k].skip())
                    {
                        int idx = anymal_->getContacts()[k].getlocalBodyIndex();

                        // check foot height to distinguish shank contact
                        // TODO: this only works for flat terrain
                        if (idx == 2 && footPos_W[0][2] < 0.022 && !footContactState_[0])
                        {
                            footContactState_[0] = true;
                            // footNormal_[0] = anymal_->getContacts()[k].getNormal().e();
                            anymal_->getContactPointVel(k, vec3);
                            footContactVel_[0] = vec3.e();
                            numFootContact_++;
                            GRF_impulse[0] = anymal_->getContacts()[k].getImpulse()->e().squaredNorm();
                        }
                        else if (idx == 4 && footPos_W[1][2] < 0.022 && !footContactState_[1])
                        {
                            footContactState_[1] = true;
                            // footNormal_[1] = anymal_->getContacts()[k].getNormal().e();
                            anymal_->getContactPointVel(k, vec3);
                            footContactVel_[1] = vec3.e();
                            numFootContact_++;
                            GRF_impulse[1] = anymal_->getContacts()[k].getImpulse()->e().squaredNorm();
                        }
                        else if (idx == 6 && footPos_W[2][2] < 0.022 && !footContactState_[2])
                        {
                            footContactState_[2] = true;
                            // footNormal_[2] = anymal_->getContacts()[k].getNormal().e();
                            anymal_->getContactPointVel(k, vec3);
                            footContactVel_[2] = vec3.e();
                            numFootContact_++;
                            GRF_impulse[2] = anymal_->getContacts()[k].getImpulse()->e().squaredNorm();
                        }
                        else if (idx == 8 && footPos_W[3][2] < 0.022 && !footContactState_[3])
                        {
                            footContactState_[3] = true;
                            // footNormal_[3] = anymal_->getContacts()[k].getNormal().e();
                            anymal_->getContactPointVel(k, vec3);
                            footContactVel_[3] = vec3.e();
                            numFootContact_++;
                            GRF_impulse[3] = anymal_->getContacts()[k].getImpulse()->e().squaredNorm();
                        }
                    }
                }
            }
        }

        void calculate_cost()
        {
            torqueCost=0, linvelCost=0, angVelCost=0, velLimitCost = 0, footClearanceCost = 0, slipCost = 0;
            leg_phase_cost=0, desiredHeightCost=0, previousActionCost = 0, orientationCost = 0, footVelCost = 0;

            double yawRateError = (gv_[5] - 0) * (gv_[5] - 0) * (4.0 + costScale_ * 5);

            torqueCost = costScale_ * 0.5 * torque.tail(8).norm() * simulation_dt_;

            // const double velErr = std::max((4.0 + costScale_ * 5) * (desired_velocity - bodyLinearVel_[0]), 0.0);
            const double velErr = std::abs(desired_velocity - bodyLinearVel_[0]);

            linvelCost = -500.0 * simulation_dt_ / (exp(velErr) + 2.0 + exp(-velErr)); // ==> min -0.25

            angVelCost = -120.0 * simulation_dt_ / (exp(yawRateError) + 2.0 + exp(-yawRateError)); // ==> min -0.15
            // angVelCost += costScale_ * std::min(0.25 * u_.segment<2>(3).squaredNorm() * simulation_dt_, 0.002) / std::min(0.3 + 3.0 * commandNorm, 1.0);

            double velLim = 0.0;
            // for (int i = 6; i < 14; i++)
            //     if (fabs(gv_(i)) > velLim)
            //         velLimitCost += costScale_ * 0.3e-2 / std::min(0.09 + 2.5 * desired_velocity, 1.0) * (std::fabs(gv_[i]) - velLim) * (std::fabs(gv_[i]) - velLim) * simulation_dt_;

            for (int i = 6; i < 14; i++)
                if (fabs(gv_(i)) > velLim)
                    velLimitCost += costScale_ * 0.02 / std::min(std::max(2.5 * desired_velocity - 4, 0.1), 1.0) * std::fabs(gv_[i]) * simulation_dt_;

            for (int i = 0; i < 4; i++) 
            {
                footVelCost += costScale_ * 1. / std::min(std::max(2.5 * desired_velocity - 4, 0.1), 1.0) * footVel_W[i][2] * footVel_W[i][2] * simulation_dt_;

                if (!footContactState_[i]) {
                    // not in contact
                    footClearanceCost += costScale_ * 15000 * pow(std::max(0.0, 0.07 - footPos_W[i][2]), 2) * footVel_W[i].e().head(2).norm() * simulation_dt_;
                    if (current_leg_phase[i] == 0)
                        // currently in swing phase ==> should not be in contact (increase reward = decrease cost)
                        leg_phase_cost -= 1;
                }
                else{
                    // in contact
                    slipCost += 1000 * (costScale_ * (0.2 * footContactVel_[i].head(2).norm())) * simulation_dt_;
                    if (current_leg_phase[i] == 1)
                        // currently in stance phase ==> should be in contact (increase reward = decrease cost)
                        leg_phase_cost -= 1;
                }
            }

            leg_phase_cost /= 4.;
            leg_phase_cost = leg_phase_cost * simulation_dt_ * 100;

            previousActionCost = 0.5 * costScale_ * (previous_action - current_action).norm() * simulation_dt_;

            Eigen::Vector3d identityRot(0,0,1);
            orientationCost = costScale_ * 100.0 * (pitch_and_yaw - identityRot).norm() * simulation_dt_;

            cost = torqueCost + linvelCost + angVelCost + footClearanceCost + velLimitCost + slipCost + previousActionCost + orientationCost + footVelCost + leg_phase_cost; //  ;

        }

        void set_leg_phase(Eigen::Ref<EigenVec> leg_phase) 
        {
            for (int i=0; i<4; i++){
                current_leg_phase[i] = leg_phase[i];
            }
        }

        void set_next_initialize_steps(int next_initialize_steps) 
        {
            next_initialize_n_state = next_initialize_steps;
        }

        // void update_initial_state(Eigen::VectorXd current_state)
        // {
        //     next_gc_init_ = current_state;
        // }

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

        void increase_cost_scale() {
            costScale_ = std::pow(costScale_, 0.997);
        }

    private:
        int gcDim_, gvDim_, nJoints_, idx;
        bool visualizable_ = false;
        raisim::ArticulatedSystem *anymal_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque;
        double terminalRewardCoeff_ = -10., velocity, desired_velocity, reward_torque_coeff, leg_work_entropy;
        double reward_velocity_coeff, reward_impulse_coeff, reward_height_coeff, reward_orientation_coeff, GRF_entropy, GRF_impulse_reward, reward_leg_work_coeff;
        double CPG_reward_GRF_coeff, CPG_reward_velocity_coeff, CPG_rewards_, CPG_reward_torque_coeff;
        double unContactPenalty = -5., height_threshold, height_violation;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_, reward_log;
        Eigen::VectorXd single_contact_impulse;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, pitch_and_yaw;
        Eigen::ArrayXd total_contact_impulse, GRF_impulse, joint_work, leg_work;
        std::set<size_t> footIndices_, contact_foot_idx;
        /////
        std::vector<raisim::Vec<3>> footPos_;
        std::vector<raisim::Vec<3>> footPos_W;
        std::vector<raisim::Vec<3>> footVel_W;
        std::vector<Eigen::Vector3d> footContactVel_;

        Eigen::Vector4d foot_idx, current_leg_phase;

        size_t numContact_;
        size_t numFootContact_;
        size_t numBodyContact_;
        size_t numBaseContact_;

        // Buffers for contact states
        std::array<bool, 4> footContactState_;

        double costScale_ = 0.3;
        double simulation_dt = 0.0025;
        double cost;
        Eigen::VectorXd previous_action, current_action;

        double torqueCost=0, linvelCost=0, angVelCost=0, velLimitCost = 0, footClearanceCost = 0, slipCost = 0, desiredHeightCost=0, previousActionCost = 0, orientationCost = 0, footVelCost = 0, leg_phase_cost=0;

        int next_initialize_n_state=0, current_n_state=0;
        Eigen::VectorXd next_gc_init_; 
        // initialize with states in previous trajectory. If the trajectory finishes during the episode, it is initialized w/ gc_init_, not next_gc_init_

    };
    // // Logging example
    // Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
    // std::cout << current_leg_phase.format(CommaInitFmt) << std::endl;
}
