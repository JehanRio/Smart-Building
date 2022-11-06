cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_COMPILE_FEATURES "g++")
project(cmake_test)

aux_source_directory(. VAR) # 搜索dir目录下所有源文件，并将结果寸尺在变量VAR中
add_executable(cv ${VAR})    # 指定使用源文件src，生成可执行程序target,${变量名}是取变量的值

include_directories(.)   # 设置包含的头文件目录  