cmake_minimum_required(VERSION 3.20.0)
project(ptoas)

# =========================================================
# [新增] 强制设置 C++17 标准 (LLVM 19 必需)
# =========================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =========================================================
# 1. 显式寻找 LLVM (这是解决 tablegen 报错的关键)
# =========================================================
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM CMake Dir: ${LLVM_CMAKE_DIR}")
message(STATUS "LLVM Include Dir: ${LLVM_INCLUDE_DIRS}")

# 将 LLVM 模块路径加入 CMake 搜索路径
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# 关键：加载 LLVM 基础宏 (这里定义了 tablegen, add_llvm_pass 等)
include(AddLLVM)
include(TableGen) # 双重保险，确保 tablegen 宏可用
include(HandleLLVMOptions) # 处理 RTTI, EH 等编译选项

# =========================================================
# 2. 寻找 MLIR
# =========================================================
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "MLIR CMake Dir: ${MLIR_CMAKE_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

# =========================================================
# 3. 配置构建环境
# =========================================================
# macOS: 移除已废弃的 -undefined suppress，避免 ld 警告（保留 -undefined dynamic_lookup 即可）
if(APPLE)
  foreach(_var CMAKE_SHARED_MODULE_CREATE_CXX_FLAGS CMAKE_MODULE_LINKER_FLAGS)
    string(REPLACE "-Wl,-undefined -Wl,suppress" "" ${_var} "${${_var}}")
    string(REPLACE "-undefined suppress" "" ${_var} "${${_var}}")
  endforeach()
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# =========================================================
# 4. 添加子目录
# =========================================================
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# 开启 Python 绑定选项
option(PTO_ENABLE_PYTHON_BINDING "Enable Python bindings" ON)

if(PTO_ENABLE_PYTHON_BINDING)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)

  # 设置 Python 扩展的输出目录，方便调试
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/pto)

  add_subdirectory(python)
endif()

# =========================================================
# 5. 安装配置 (生成PTOASConfig.cmake)
# 将 include/PTO目录下的头文件安装到 <prefix>/include/PTO
install(DIRECTORY include/PTO
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT PTOAS_Development
        FILES_MATCHING PATTERN "*.h" PATTERN "*.td"
)
# 安装构建目录下的 .inc 生成文件
install(DIRECTORY ${PROJECT_BINARY_DIR}/include/PTO
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT PTOAS_Development
        FILES_MATCHING PATTERN "*.inc"
)

# 6. 定义 CMake 配置文件安装位置 (例如 lib/cmake/PTOAS)
set(PTOAS_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/PTOAS")

# 7. 导出 Targets (PTOASTargets.cmake) 供 find_package 使用
install(EXPORT PTOASTargets
        FILE PTOASTargets.cmake
        NAMESPACE PTOAS::
        DESTINATION ${PTOAS_CMAKE_INSTALL_DIR}
)

# 8. 生成 Config 文件 (PTOASConfig.cmake)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PTOASConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/PTOASConfig.cmake"
    INSTALL_DESTINATION ${PTOAS_CMAKE_INSTALL_DIR}
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

# 9. 安装 Config 文件
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/PTOASConfig.cmake"
    DESTINATION ${PTOAS_CMAKE_INSTALL_DIR}
)