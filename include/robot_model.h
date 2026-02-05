/**
 * * * @description:
 * * * @filename: robot_model.h
 * * * @author: wangxurui
 * * * @date: 2025-04-08 12:45:56
 **/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

#include <grid_map_msgs/GridMap.h>
#include <grid_map_core/grid_map_core.hpp>

using namespace std;

class robot_model
{
private:
  string mapfile_;
  grid_map::GridMap robot_gridmap_;

public:
  robot_model(string &mapfile);
  ~robot_model();
  loadCsvToGridMap();
};

robot_model::robot_model(string &mapfile) : mapfile_(mapfile)
{
  loadCsvToGridMap();
}

robot_model::~robot_model()
{
}

void robot_model::loadCsvToGridMap()
{
  std::ifstream file(mapfile_);
  std::string line;
  std::vector<std::vector<double>> data;

  // 读取 CSV 文件
  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    std::string value;
    std::vector<double> row;

    while (std::getline(ss, value, ','))
    {
      row.push_back(std::stod(value)); // 转换为 double
    }
    data.push_back(row);
  }

  // 将数据转换为 Eigen 矩阵
  int rows = data.size();
  int cols = data[0].size();
  Eigen::MatrixXd heightMap(rows, cols);

  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      heightMap(i, j) = data[i][j];
    }
  }

  // 将高度图添加到 GridMap
  robot_gridmap_.add("height", heightMap);
}
