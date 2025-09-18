// test_grid_map.cpp
// Author: MarineRock10
// License: MIT
// Version: 1.0.0

#include "grid_map.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

// Test class that exposes protected members of GridMap for testing
class GridMapTester : public grid_map::GridMap {
public:
    using grid_map::GridMap::isOccupied;
    using grid_map::GridMap::posToIndex;
    using grid_map::GridMap::indexToPos;
    using grid_map::GridMap::updateESDF;
    using grid_map::GridMap::getDistanceAndGradient;
};

// Visualize distance field and gradient directions
void visualizeWithOpenCV(
    const GridMapTester& map,
    const std::string& window_name = "Grid Map & Gradient Visualization"
) {
    const int scale = 4;
    const int arrow_spacing = 1;
    const double max_dist_for_display = 2.0;
    
    const Eigen::Vector2i voxel_num = map.getVoxelNum();
    const int rows = voxel_num.y() * scale;
    const int cols = voxel_num.x() * scale;

    // Create combined image
    cv::Mat combined_img(rows, cols * 2, CV_8UC3, cv::Scalar(240, 240, 240));
    cv::Mat dist_img(rows, cols, CV_8UC3, cv::Scalar(240, 240, 240));
    cv::Mat grad_img(rows, cols, CV_8UC3, cv::Scalar(240, 240, 240));

    // Calculate maximum gradient magnitude
    double max_grad_mag = 0.0;
    for (int x = 0; x < voxel_num.x(); x += arrow_spacing) {
        for (int y = 0; y < voxel_num.y(); y += arrow_spacing) {
            Eigen::Vector2i idx(x, y);
            Eigen::Vector2d pos;
            map.indexToPos(idx, pos);
            double dist = map.getDistance(pos);
            if (std::abs(dist) < max_dist_for_display) {
                Eigen::Vector2d grad;
                if (map.getDistanceAndGradient(pos, dist, grad)) {
                    max_grad_mag = std::max(max_grad_mag, grad.norm());
                }
            }
        }
    }

    // Draw distance field (left side)
    for (int x = 0; x < voxel_num.x(); ++x) {
        for (int y = 0; y < voxel_num.y(); ++y) {
            Eigen::Vector2i idx(x, y);
            Eigen::Vector2d pos;
            map.indexToPos(idx, pos);
            double dist = map.getDistance(pos);
            
            cv::Scalar color;
            if (std::abs(dist) < 1e-6) {
                color = cv::Scalar(0, 255, 255); // Boundary: yellow
            } else if (dist > 0) {
                // Free space: blue gradient
                double norm_dist = std::min(dist, max_dist_for_display) / max_dist_for_display;
                int blue = 255;
                int green = static_cast<int>((1.0 - norm_dist) * 200);
                color = cv::Scalar(blue, green, 0);
            } else {
                // Occupied space: red gradient
                double norm_dist = std::min(-dist, max_dist_for_display) / max_dist_for_display;
                int red = 255;
                int green = static_cast<int>((1.0 - norm_dist) * 200);
                color = cv::Scalar(0, green, red);
            }
            
            cv::rectangle(dist_img, 
                cv::Point(y*scale, x*scale), 
                cv::Point((y+1)*scale-1, (x+1)*scale-1), 
                color, -1);
        }
    }

    // Draw gradient arrows (right side)
    for (int x = 0; x < voxel_num.x(); x += arrow_spacing) {
        for (int y = 0; y < voxel_num.y(); y += arrow_spacing) {
            Eigen::Vector2i idx(x, y);
            Eigen::Vector2d pos;
            map.indexToPos(idx, pos);
            double dist = map.getDistance(pos);
            
            if (std::abs(dist) < max_dist_for_display) {
                Eigen::Vector2d grad;
                if (map.getDistanceAndGradient(pos, dist, grad)) {
                    double mag = grad.norm();
                    if (mag > 1e-3) {
                        cv::Point center(y*scale + scale/2, x*scale + scale/2);
                        cv::Point end(
                            center.x + static_cast<int>(grad.y() * scale * 0.8 / mag),
                            center.y + static_cast<int>(grad.x() * scale * 0.8 / mag)
                        );
                        
                        cv::Scalar color;
                        if (std::abs(dist) < 1e-6) {
                            color = cv::Scalar(0, 255, 255);
                        } else if (dist < 0) {
                            color = cv::Scalar(255, 0, 255);
                        } else {
                            double norm_mag = max_grad_mag > 0 ? mag / max_grad_mag : 0;
                            int g = 255;
                            int r = static_cast<int>(norm_mag * 255);
                            color = cv::Scalar(0, g, r);
                        }
                        
                        cv::arrowedLine(grad_img, center, end, color, 1, cv::LINE_AA, 0, 0.3);
                    }
                }
            }
        }
    }

    // Combine images and add labels
    dist_img.copyTo(combined_img(cv::Rect(0, 0, cols, rows)));
    grad_img.copyTo(combined_img(cv::Rect(cols, 0, cols, rows)));

    cv::putText(combined_img, "Distance Field (Blue=free, Red=occupied, Yellow=boundary)", 
                cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(combined_img, "Gradient Direction (Purple=inside, Green=free, Yellow=boundary)", 
                cv::Point(cols + 10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    std::string dist_legend = "Distance range: [-" + std::to_string(max_dist_for_display) + 
                             "m, " + std::to_string(max_dist_for_display) + "m]";
    cv::putText(combined_img, dist_legend, cv::Point(10, rows - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    cv::imshow(window_name, combined_img);
    cv::waitKey(0);
}

// Verify gradient computation using finite differences
void verifyGradientComputation(
    const GridMapTester& map,
    const Eigen::Vector2d& test_point,
    double epsilon = 1e-4
) {
    if (!map.isInsideMap(test_point)) {
        std::cerr << "Test point is outside map boundary!" << std::endl;
        return;
    }

    // Analytic gradient
    double dist;
    Eigen::Vector2d grad_analytic;
    bool success = map.getDistanceAndGradient(test_point, dist, grad_analytic);
    if (!success) {
        std::cerr << "Failed to compute analytic gradient!" << std::endl;
        return;
    }

    // Numeric gradient (central difference)
    Eigen::Vector2d grad_numeric;
    
    // x-direction
    Eigen::Vector2d pt_plus_x = test_point + Eigen::Vector2d(epsilon, 0);
    Eigen::Vector2d pt_minus_x = test_point - Eigen::Vector2d(epsilon, 0);
    double dist_plus_x = map.getDistance(pt_plus_x);
    double dist_minus_x = map.getDistance(pt_minus_x);
    grad_numeric.x() = (dist_plus_x - dist_minus_x) / (2 * epsilon);

    // y-direction
    Eigen::Vector2d pt_plus_y = test_point + Eigen::Vector2d(0, epsilon);
    Eigen::Vector2d pt_minus_y = test_point - Eigen::Vector2d(0, epsilon);
    double dist_plus_y = map.getDistance(pt_plus_y);
    double dist_minus_y = map.getDistance(pt_minus_y);
    grad_numeric.y() = (dist_plus_y - dist_minus_y) / (2 * epsilon);

    // Calculate relative error
    Eigen::Vector2d diff = grad_analytic - grad_numeric;
    double rel_error = diff.norm() / (grad_numeric.norm() + 1e-6);

    std::cout << "\n=== Gradient Verification at (" << test_point.x() << ", " << test_point.y() << ") ===\n";
    std::cout << "Analytic gradient: (" << grad_analytic.x() << ", " << grad_analytic.y() << ")\n";
    std::cout << "Numeric gradient:  (" << grad_numeric.x() << ", " << grad_numeric.y() << ")\n";
    std::cout << "Relative error: " << rel_error << "\n";

    if (rel_error > 1e-2) {
        std::cerr << "Warning: High gradient computation error detected!" << std::endl;
    }
}

// Benchmark ESDF computation performance
void benchmarkESDF(GridMapTester& map) {
    auto start = std::chrono::high_resolution_clock::now();
    map.updateESDF();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "ESDF computation time: " << elapsed.count() << " seconds\n";
}

// Test coordinate conversion functions
void testCoordinateConversion(const GridMapTester& map) {
    std::cout << "\n=== Coordinate Conversion Tests ===\n";
    
    Eigen::Vector2d test_points[] = {
        {0.0, 0.0},
        {-10.0, -10.0},
        {10.0, 10.0},
        {5.5, -3.2}
    };

    for (const auto& pt : test_points) {
        Eigen::Vector2i idx;
        map.posToIndex(pt, idx);
        
        Eigen::Vector2d reconstructed_pt;
        map.indexToPos(idx, reconstructed_pt);
        
        std::cout << "World pos (" << pt.x() << ", " << pt.y() << ") -> "
                  << "Index [" << idx.transpose() << "] -> "
                  << "Reconstructed pos (" << reconstructed_pt.x() << ", " << reconstructed_pt.y() << ")\n";
        
        double error = (pt - reconstructed_pt).norm();
        if (error > 1e-6) {
            std::cerr << "Warning: High conversion error: " << error << std::endl;
        }
    }
}

// Test collision detection functionality
void testCollisionChecking(const GridMapTester& map) {
    std::cout << "\n=== Collision Check Tests ===\n";
    Eigen::Vector2d test_points[] = {
        {0.0, 0.0},
        {-9.5, -9.5},
        {9.5, 9.5},
        {5.0, 5.0},
        {-5.0, -5.0}
    };

    for (const auto& pt : test_points) {
        bool collision = map.isCollision(pt);
        double dist = map.getDistance(pt);
        std::cout << "Point (" << pt.x() << ", " << pt.y() << "): " 
                  << (collision ? "COLLISION" : "SAFE")
                  << " | Distance: " << dist << "m\n";
        
        if (dist < 0) {
            std::cerr << "Warning: Negative distance value detected!" << std::endl;
        }
    }
}

// Test line-of-sight functionality
void testLineOfSight(GridMapTester& map) {
    std::cout << "\n=== Line of Sight Tests ===\n";
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> test_lines = {
        {{-5, -5}, {5, 5}},
        {{-8, 0}, {8, 0}},
        {{0, -8}, {0, 8}},
        {{-7, -7}, {7, 7}}
    };

    for (const auto& line : test_lines) {
        bool blocked = map.isLineOccupancy(line.first, line.second);
        std::cout << "Line from (" << line.first.x() << ", " << line.first.y() 
                  << ") to (" << line.second.x() << ", " << line.second.y() << "): "
                  << (blocked ? "BLOCKED" : "CLEAR") << "\n";
    }
}

// Test map boundary functionality
void testMapBoundary(const GridMapTester& map) {
    std::cout << "\n=== Map Boundary Tests ===\n";
    Eigen::Vector2d boundary_points[] = {
        {10.0, 10.0},
        {-10.0, -10.0},
        {10.0, -10.0},
        {-10.0, 10.0},
        {11.0, 0.0},
        {0.0, -11.0}
    };

    for (const auto& pt : boundary_points) {
        double dist = map.getDistance(pt);
        bool is_inside = (dist < 1e+9);
        std::cout << "Point (" << pt.x() << ", " << pt.y() << "): "
                  << (is_inside ? "INSIDE" : "OUTSIDE") << " map | "
                  << "Distance: " << dist << "m\n";
    }
}

int main() {
    // 1. Map initialization test
    std::cout << "\n===== 1. Map Initialization Test =====" << std::endl;
    GridMapTester map;
    map.init(20.0, 20.0, 0.1);
    std::cout << "Map initialized with resolution: " << 0.1 << "m\n";
    std::cout << "Voxel numbers: " << map.getVoxelNum().transpose() << "\n";

    // 2. Obstacle setting test
    std::cout << "\n===== 2. Obstacle Setting Test =====" << std::endl;
    grid_map::RowMatrixXi occupancy = grid_map::RowMatrixXi::Zero(200, 200);
    
    // Add L-shaped wall obstacles
    occupancy.block(70, 70, 60, 5).setOnes();
    occupancy.block(100, 50, 5, 80).setOnes();
    std::cout << "Obstacles set in the map (L-shaped wall)\n";

    // 3. ESDF computation test
    std::cout << "\n===== 3. ESDF Computation Test =====" << std::endl;
    map.setMap(occupancy);
    benchmarkESDF(map);

    // 4. Function test suite
    std::cout << "\n===== 4. Function Test Suite =====" << std::endl;
    testCoordinateConversion(map);
    testCollisionChecking(map);
    testLineOfSight(map);
    testMapBoundary(map);

    // 5. Gradient verification
    std::cout << "\n===== 5. Gradient Verification =====" << std::endl;
    verifyGradientComputation(map, Eigen::Vector2d(0.0, 0.0));
    verifyGradientComputation(map, Eigen::Vector2d(5.0, 5.0));
    verifyGradientComputation(map, Eigen::Vector2d(7.0, 7.0));

    // 6. Visualization
    std::cout << "\n===== 6. Visualization =====" << std::endl;
    visualizeWithOpenCV(map);

    return 0;
}