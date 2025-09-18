// test_opt.cpp
// Author: MarineRock10
// License: MIT
// Version: 1.0.0

#include "astar.hpp"
#include "grid_map.hpp"
#include "traj_opt.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>

// Visualize path planning results
void visualize(const grid_map::GridMap& map,
              const std::vector<Eigen::Vector2d>& path,
              const Eigen::Vector2d& start,
              const Eigen::Vector2d& goal,
              double viz_size = 30.0,
              const std::string& window_name = "Path Planning");

int main() {
    // ==================== 1. Initialize Local Map ====================
    auto map = std::make_shared<grid_map::GridMap>();
    const double map_size = 20.0;
    const double resolution = 0.1;
    map->init(map_size, map_size, resolution);

    // ==================== 2. Set Obstacles ====================
    grid_map::RowMatrixXi occupancy = grid_map::RowMatrixXi::Zero(200, 200);
    
    // Horizontal wall (x: -5m~5m, y: 0m)
    for (int x = 50; x < 200; ++x) {
        for (int y = 95; y < 100; ++y) {
            occupancy(x, y) = 1;
        }
    }
    
    // Vertical wall (x < 8m, y = 15.5m)
    for (int x = 0; x < 80; ++x) {
        for (int y = 155; y < 160; ++y) {
            occupancy(x, y) = 1;
        }
    }
    
    // Additional vertical wall
    for (int x = 0; x < 80; ++x) {
        for (int y = 35; y < 40; ++y) {
            occupancy(x, y) = 1;
        }
    }
    
    map->setMap(occupancy);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // ==================== 3. Define Start and Goal ====================
    const Eigen::Vector2d start(-6.0, -8.0);
    const Eigen::Vector2d goal(6.0, 6.0);

    // ==================== 4. A* Global Search ====================
    path_planning::AStar astar(*map, 0.2); // 0.2m safety threshold
    auto astar_traj = astar.planWithPostProcessing(start, goal, 5000); // 5s timeout
    
    if (astar_traj.optimized_path.empty()) {
        std::cerr << "A* planning failed!" << std::endl;
        return -1;
    }

    std::cout << "=== A* Planning Results ===" << std::endl;
    std::cout << "Optimized path points: " << astar_traj.optimized_path.size() << std::endl;
    std::cout << "Total length: " << astar_traj.total_length << " m" << std::endl;
    std::cout << "Total time: " << astar_traj.total_time << " s" << std::endl;
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();

    visualize(*map, astar_traj.optimized_path, start, goal, 40.0, "A* Optimized Path");

    // ==================== 5. Trajectory Optimization ====================
    TrajOpt::TrajectoryParams params;
    params.piece_len = astar_traj.total_length / astar_traj.total_time;
    params.total_time = astar_traj.total_time;
    params.total_len = astar_traj.total_length;

    TrajOpt::TrajectoryOptimizer optimizer(
        map,
        astar_traj.optimized_path,
        params
    );

    if (!optimizer.plan()) {
        std::cerr << "Trajectory optimization failed!" << std::endl;
        return -1;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // ==================== 6. Evaluate Optimization Results ====================
    auto metrics = optimizer.evaluateTrajectory();
    std::cout << "\n=== Optimization Metrics ===" << std::endl;
    std::cout << "Max velocity: " << metrics.max_velocity << " m/s" << std::endl;
    std::cout << "Min clearance: " << metrics.min_clearance << " m" << std::endl;
    std::cout << "Path deviation: " << metrics.path_deviation << " m" << std::endl;
    std::cout << "Trajectory energy: " << metrics.trajectory_energy << std::endl;
    std::cout << "Optimization time: " << duration.count() << " ms" << std::endl;

    // ==================== 7. Visualize Optimized Trajectory ====================
    auto opt_path = optimizer.sampleTrajectory(0.1); // Sample every 0.1s
    visualize(*map, opt_path, start, goal, 40.0, "Optimized Trajectory");

    return 0;
}

// Visualization function implementation
void visualize(const grid_map::GridMap& map,
              const std::vector<Eigen::Vector2d>& path,
              const Eigen::Vector2d& start,
              const Eigen::Vector2d& goal,
              double viz_size,
              const std::string& window_name) 
{
    const double resolution = map.getResolution();
    const int scale = 2; // Visualization scaling factor
    const int viz_pixels = static_cast<int>(viz_size / resolution) * scale;
    
    // Create white background image
    cv::Mat img(viz_pixels, viz_pixels, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // World to pixel coordinate conversion
    auto worldToPixel = [&](const Eigen::Vector2d& pt) {
        int x = viz_pixels/2 + pt.x() / resolution * scale;
        int y = viz_pixels/2 - pt.y() / resolution * scale; // Flip Y-axis
        return cv::Point(x, y);
    };

    // Draw map boundary (red rectangle)
    Eigen::Vector2d map_min = -map.getMapSize()/2.0;
    Eigen::Vector2d map_max = map.getMapSize()/2.0;
    cv::rectangle(img, worldToPixel(map_min), worldToPixel(map_max), 
                 cv::Scalar(0, 0, 255), 2);

    // Draw obstacles (black squares)
    for (int x = 0; x < map.getVoxelNum().x(); ++x) {
        for (int y = 0; y < map.getVoxelNum().y(); ++y) {
            if (map.isOccupied(Eigen::Vector2i(x, y))) {
                Eigen::Vector2d pos;
                map.indexToPos(Eigen::Vector2i(x, y), pos);
                cv::rectangle(img, 
                    worldToPixel(pos) - cv::Point(scale/2, scale/2),
                    worldToPixel(pos) + cv::Point(scale/2, scale/2), 
                    cv::Scalar(0, 0, 0), -1);
            }
        }
    }

    // Draw path (green lines)
    for (size_t i = 1; i < path.size(); ++i) {
        cv::line(img, worldToPixel(path[i-1]), worldToPixel(path[i]), 
                cv::Scalar(0, 255, 0), 2);
    }

    // Mark start (red) and goal (blue) points
    cv::circle(img, worldToPixel(start), scale*2, cv::Scalar(0, 0, 255), -1);
    cv::circle(img, worldToPixel(goal), scale*2, cv::Scalar(255, 0, 0), -1);
    
    // Add labels
    cv::putText(img, "Start", worldToPixel(start) + cv::Point(scale*3, 0), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    cv::putText(img, "Goal", worldToPixel(goal) + cv::Point(scale*3, 0), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

    // Display image
    cv::imshow(window_name, img);
    cv::waitKey(0);
}