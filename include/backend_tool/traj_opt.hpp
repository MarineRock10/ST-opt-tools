// traj_opt.hpp
// Author: MarineRock10
// License: MIT
// Version: 1.0.0

#ifndef TRAJ_OPT_HPP
#define TRAJ_OPT_HPP

#include "SplineTrajectory.hpp"
#include "grid_map.hpp"
#include <lbfgs.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

namespace TrajOpt {

struct TrajectoryParams {
    double total_len = 10;
    double total_time = 10;
    double piece_len = 4;
    double rho_v = 10000;           // Velocity penalty weight
    double rho_collision = 100000;  // Collision penalty weight
    double rho_T = 100;             // Time penalty weight
    double rho_energy = 100;        // Energy (smoothness) penalty weight
    double max_v = 1.0;             // Maximum velocity
    double safe_threshold = 0.5;    // Safety distance threshold
    
    int int_K = 32;                 // Integration sample points
    int mem_size = 256;             // L-BFGS memory size
    int past = 3;                   // L-BFGS parameter
    double g_epsilon = 1e-6;        // Gradient convergence threshold
    double min_step = 1e-32;        // Minimum step size
    double delta = 1e-5;            // Function change convergence threshold
    int max_iter = 10000;           // Maximum iterations
};

class TrajectoryOptimizer {
public:
    using Vector2d = Eigen::Vector2d;
    using Vector3d = Eigen::Vector3d;
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using PPoly2D = SplineTrajectory::PPolyND<2>;
    using CubicSpline2D = SplineTrajectory::CubicSpline2D;

    TrajectoryOptimizer(std::shared_ptr<grid_map::GridMap> map,
                       const std::vector<Eigen::Vector2d>& astar_path,
                       const TrajectoryParams& params = TrajectoryParams())
        : map_(map), astar_path_(astar_path), params_(params), in_opt_(false) 
    {
        preprocessAstarPath();
    }

    bool plan() {
        if (astar_path_.size() < 2) return false;
        
        // Prepare initial conditions
        Eigen::Matrix2d init_cond, end_cond;
        init_cond.col(0) = astar_path_.front();
        end_cond.col(0) = astar_path_.back();
        init_cond.col(1) = (astar_path_[1] - astar_path_[0]).normalized() * 0.1;
        end_cond.col(1) = (astar_path_.back() - astar_path_[astar_path_.size()-2]).normalized() * 0.1;

        double total_len = params_.total_len;
        int piece_num = (int)(total_len / params_.piece_len);
        if (piece_num < 2) piece_num = 2;
        params_.piece_len = total_len / piece_num;
        
        // Sample intermediate control points from A* path
        Eigen::MatrixXd inner_pos(2, piece_num-1);
        std::vector<Eigen::Vector2d> inner_pos_node;
        
        double step_len = total_len / piece_num;
        double accumulated_len = 0.0;
        
        for (int i = 1; i < piece_num; ++i) {
            double target_len = i * step_len;
            while (accumulated_len < target_len && current_segment_ < astar_path_.size() - 1) {
                double seg_len = (astar_path_[current_segment_ + 1] - astar_path_[current_segment_]).norm();
                if (accumulated_len + seg_len >= target_len) {
                    double ratio = (target_len - accumulated_len) / seg_len;
                    Eigen::Vector2d point = astar_path_[current_segment_] + 
                                          ratio * (astar_path_[current_segment_ + 1] - astar_path_[current_segment_]);
                    inner_pos_node.push_back(point);
                    break;
                }
                accumulated_len += seg_len;
                current_segment_++;
            }
        }
        
        inner_pos.resize(2, inner_pos_node.size());
        for (size_t i = 0; i < inner_pos_node.size(); i++) {
            inner_pos.col(i) = inner_pos_node[i];
        }

        double total_time = params_.total_time;
        int result = optimizeSE2Traj(init_cond, inner_pos, end_cond, total_time);
        return result;
    }

    PPoly2D getOptimizedTrajectory() const { return trajectory_; }

    struct TrajectoryMetrics {
        double max_velocity;
        double min_clearance;
        double total_time;
        double trajectory_energy;
        double path_deviation;
    };
    
    TrajectoryMetrics evaluateTrajectory() const {
        TrajectoryMetrics metrics;
        metrics.max_velocity = 0.0;
        metrics.min_clearance = std::numeric_limits<double>::max();
        metrics.total_time = trajectory_.getDuration();
        metrics.trajectory_energy = 0.0;
        metrics.path_deviation = 0.0;

        if (!trajectory_.isInitialized()) return metrics;

        const double dt = 0.01;
        int sample_count = 0;
        for (double t = 0.0; t < metrics.total_time; t += dt) {
            Vector2d pos = trajectory_.evaluate(t, 0);
            Vector2d vel = trajectory_.evaluate(t, 1);
            Vector2d acc = trajectory_.evaluate(t, 2);
            
            metrics.trajectory_energy += acc.squaredNorm() * dt;
            metrics.max_velocity = std::max(metrics.max_velocity, vel.norm());
            metrics.min_clearance = std::min(metrics.min_clearance, map_->getDistance(pos));
            
            double min_dist = std::numeric_limits<double>::max();
            for (const auto& astar_pt : astar_path_) {
                min_dist = std::min(min_dist, (pos - astar_pt).norm());
            }
            metrics.path_deviation += min_dist;
            sample_count++;
        }
        
        if (sample_count > 0) {
            metrics.path_deviation /= sample_count;
        }
        
        return metrics;
    }

    std::vector<Eigen::Vector2d> sampleTrajectory(double dt = 0.1) const {
        std::vector<Eigen::Vector2d> path;
        if (!trajectory_.isInitialized()) return path;
        
        const double total_time = trajectory_.getDuration();
        for (double t = 0.0; t <= total_time; t += dt) {
            path.push_back(trajectory_.evaluate(t, 0));
        }
        
        if (!path.empty() && path.back() != trajectory_.evaluate(total_time, 0)) {
            path.push_back(trajectory_.evaluate(total_time, 0));
        }
        
        return path;
    }

private:
    void preprocessAstarPath() {
        // Prepare A* path for fast nearest neighbor queries
    }

    int optimizeSE2Traj(const MatrixXd& initPos, const MatrixXd& innerPtsPos,
                        const MatrixXd& endPos, double totalTime) {
        in_opt_ = true;
        piece_pos_ = innerPtsPos.cols() + 1;

        init_pos_ = initPos;
        end_pos_ = endPos;

        int variable_num = 2 * (piece_pos_ - 1) + 1;
        dim_T = 1;

        Eigen::VectorXd x;
        x.resize(variable_num);

        double& tau = x(0);
        Eigen::Map<Eigen::MatrixXd> Ppos(x.data() + dim_T, 2, piece_pos_ - 1);
        
        tau = logC2(totalTime);
        Ppos = innerPtsPos;

        Eigen::VectorXd Tpos;
        Tpos.resize(piece_pos_);
        calTfromTauUni(tau, Tpos);

        generateTrajectory(initPos, endPos, Ppos, Tpos);

        auto metrics = evaluateTrajectory();
        std::cout << "Initial Trajectory:" << std::endl;
        std::cout << "Max velocity: " << metrics.max_velocity << " m/s" << std::endl;
        std::cout << "Min clearance: " << metrics.min_clearance << " m" << std::endl;
        std::cout << "Path deviation: " << metrics.path_deviation << " m" << std::endl;

        lbfgs::lbfgs_parameter_t lbfgs_params;
        lbfgs_params.mem_size = params_.mem_size;
        lbfgs_params.past = params_.past;
        lbfgs_params.g_epsilon = params_.g_epsilon;
        lbfgs_params.min_step = params_.min_step;
        lbfgs_params.delta = params_.delta;
        lbfgs_params.max_iterations = params_.max_iter;

        double final_cost;
        int result = lbfgs::lbfgs_optimize(
            x, final_cost,
            [](void* instance, const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
                return static_cast<TrajectoryOptimizer*>(instance)->costFunction(x, grad);
            },
            nullptr, nullptr, this, lbfgs_params);

        std::cout << "Optimization finished with result: " << result << std::endl;
        std::cout << "Final cost: " << final_cost << std::endl;

        in_opt_ = false;
        return result;
    }

    double costFunction(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        double cost = 0.0;

        const double& tau = x(0);
        double& grad_tau = grad(0);
        Eigen::Map<const Eigen::MatrixXd> Ppos(x.data() + dim_T, 2, piece_pos_ - 1);
        Eigen::Map<Eigen::MatrixXd> gradPpos(grad.data() + dim_T, 2, piece_pos_ - 1);
     
        Eigen::VectorXd Tpos;
        Tpos.resize(piece_pos_);
        calTfromTauUni(tau, Tpos);
        generateTrajectory(init_pos_, end_pos_, Ppos, Tpos);

        double constrain_cost = 0.0;
        Eigen::MatrixXd gdCpos_constrain;
        Eigen::VectorXd gdTpos_constrain;
        calculateConstraintCostGrad(trajectory_, constrain_cost, gdCpos_constrain, gdTpos_constrain);
        
        Eigen::MatrixXd gradPpos_constrain;
        Eigen::VectorXd gradTpos_constrain;
        calGradCTtoQT(gdCpos_constrain, gdTpos_constrain, gradPpos_constrain, gradTpos_constrain);
        
        double energy = cubic_spline_.getEnergy();
        double energy_cost = params_.rho_energy * energy;
        
        CubicSpline2D::MatrixType gradP_energy = cubic_spline_.getEnergyGradInnerP();
        Eigen::VectorXd gradT_energy = cubic_spline_.getEnergyGradTimes();
        
        gradPpos = gradPpos_constrain + params_.rho_energy * gradP_energy.transpose();
        Eigen::VectorXd gradTpos_total = gradTpos_constrain + params_.rho_energy * gradT_energy;

        double tau_cost = params_.rho_T * expC2(tau);
        double grad_Tsum = params_.rho_T + gradTpos_total.sum() / piece_pos_;
        grad_tau = grad_Tsum * getTtoTauGrad(tau);

        cost = constrain_cost + energy_cost + tau_cost;
        return cost;
    }

    void calculateConstraintCostGrad(
        PPoly2D& traj,
        double& cost,
        Eigen::MatrixXd& gdCpos,
        Eigen::VectorXd& gdTpos)
    {
        cost = 0.0;
        double v_cost = 0.0;
        double occ_cost = 0.0;
        double path_cost = 0.0;
        
        const int N = traj.getNumSegments();
        gdCpos.resize(4 * N, 2);
        gdCpos.setZero();
        gdTpos.resize(N);
        gdTpos.setZero();

        const auto& breaks = traj.getBreakpoints();
        const MatrixXd& coeffs = traj.getCoefficients();

        Eigen::Vector2d pos, vel, acc;
        double grad_time = 0.0;
        Eigen::Vector2d grad_p = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_v = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_sdf = Eigen::Vector2d::Zero();
        Eigen::Matrix<double, 4, 1> beta0, beta1, beta2;
        double s1, s2, s3;
        double step, alpha, omg;

        for (int i = 0; i < N; ++i) {
            const Eigen::Matrix<double, 4, 2>& c = coeffs.block<4, 2>(i * 4, 0);
            step = (breaks[i+1] - breaks[i]) / params_.int_K;
            s1 = 0.0;

            for (int j = 0; j <= params_.int_K; ++j) {
                alpha = 1.0 / params_.int_K * j;
                grad_p.setZero();
                grad_v.setZero();
                grad_sdf.setZero();
                grad_time = 0.0;

                s2 = s1 * s1;
                s3 = s2 * s1;
                beta0 << 1.0, s1, s2, s3;
                beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2;
                beta2 << 0.0, 0.0, 2.0, 6.0 * s1;
                pos = c.transpose() * beta0;
                vel = c.transpose() * beta1;
                acc = c.transpose() * beta2;

                omg = (j == 0 || j == params_.int_K) ? 0.5 : 1.0;

                // 1. Velocity constraint
                double vxy_snorm = vel.squaredNorm();
                double vViola = vxy_snorm - params_.max_v * params_.max_v;
                if (vViola > 0) {
                    grad_v += params_.rho_v * 6 * vViola * vViola * vel;
                    double cost_v = params_.rho_v * vViola * vViola * vViola;
                    cost += cost_v * omg * step;
                    v_cost += cost_v * omg * step;
                    grad_time += omg * (cost_v / params_.int_K + step * alpha * grad_v.dot(acc));
                }

                // 2. Collision constraint
                double sdf_value;
                map_->getDistanceAndGradient(pos, sdf_value, grad_sdf);
                double cViola = params_.safe_threshold - sdf_value;
                if (cViola > 0 && sdf_value < 5) {
                    double penalty;
                    Eigen::Vector2d grad_pc;
                    
                    if (cViola < 0.1) {
                        penalty = cViola * cViola;
                        grad_pc = -2.0 * cViola * grad_sdf;
                    } else {
                        penalty = cViola;
                        grad_pc = -grad_sdf;
                    }
                    
                    double cost_c = params_.rho_collision * penalty;
                    Eigen::Vector2d grad_pc_scaled = params_.rho_collision * grad_pc;
                    
                    cost += cost_c * omg * step;
                    occ_cost += cost_c * omg * step;
                    grad_time += omg * (cost_c / params_.int_K + step * alpha * grad_pc_scaled.dot(vel));
                    grad_p += grad_pc_scaled;
                }

                gdCpos.block<4, 2>(i * 4, 0) += 
                    (beta0 * grad_p.transpose() + beta1 * grad_v.transpose()) * omg * step;
                gdTpos(i) += grad_time;
                
                s1 += step;
            }
        }

        std::cout << "Cost breakdown - Vel: " << v_cost << ", Coll: " << occ_cost << std::endl;
    }

    void calGradCTtoQT(
        const Eigen::MatrixXd& gdCpos,
        const Eigen::VectorXd& gdTpos,
        Eigen::MatrixXd& gradPpos,
        Eigen::VectorXd& gradTpos_out)
    {
        CubicSpline2D::MatrixType gdC_typed = gdCpos;
        CubicSpline2D::MatrixType gradByPoints;
        Eigen::VectorXd gradByTimes;
        
        cubic_spline_.propagateGrad(gdC_typed, gdTpos, gradByPoints, gradByTimes);
        
        gradPpos = gradByPoints.transpose();
        gradTpos_out = gradByTimes;
        
        std::cout << "Gradient norm - Ppos: " << gradPpos.norm() 
                << ", Tpos: " << gradTpos_out.norm() << std::endl;
    }

    void generateTrajectory(const MatrixXd& initPos, const MatrixXd& endPos,
                          const MatrixXd& innerPts, Eigen::VectorXd Tpos) {
        std::vector<double> times;
        times.reserve(piece_pos_ + 1);
        
        double t = 0;
        for (int i = 0; i < Tpos.size(); ++i) {
            times.push_back(t);
            t += Tpos(i);
        }
        times.push_back(t);

        SplineTrajectory::SplineVector2D waypoints;
        waypoints.push_back(initPos.col(0));
        for (int i = 0; i < innerPts.cols(); ++i) {
            waypoints.push_back(innerPts.col(i));
        }
        waypoints.push_back(endPos.col(0));

        SplineTrajectory::BoundaryConditions<2> bc;
        bc.start_velocity = initPos.col(1);
        bc.end_velocity = endPos.col(1);

        cubic_spline_.update(times, waypoints, bc);
        trajectory_ = cubic_spline_.getTrajectory();
    }

    double calculatePathLength(const std::vector<Eigen::Vector2d>& path) {
        double length = 0.0;
        for (size_t i = 1; i < path.size(); ++i) {
            length += (path[i] - path[i-1]).norm();
        }
        return length;
    }

    inline double logC2(const double& T) {
        return T > 1.0 ? (sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T - 1.0));
    }

    inline void calTfromTauUni(const double& tau, Eigen::VectorXd& T) {
        T.setConstant(expC2(tau) / T.size());
    }

    inline double expC2(const double& tau) {
        return tau > 0.0 ? ((0.5 * tau + 1.0) * tau + 1.0) : 1.0 / ((0.5 * tau - 1.0) * tau + 1.0);
    }

    inline double getTtoTauGrad(const double& tau) {
        if (tau > 0)
            return tau + 1.0;
        else {
            double denSqrt = (0.5 * tau - 1.0) * tau + 1.0;
            return (1.0 - tau) / (denSqrt * denSqrt);
        } 
    }

    std::shared_ptr<grid_map::GridMap> map_;
    std::vector<Eigen::Vector2d> astar_path_;
    TrajectoryParams params_;
    bool in_opt_;
    int piece_pos_;
    int dim_T;
    MatrixXd init_pos_;
    MatrixXd end_pos_;
    PPoly2D trajectory_;
    CubicSpline2D cubic_spline_; 
    int current_segment_ = 0;
};

} // namespace TrajOpt

#endif // TRAJ_OPT_HPP