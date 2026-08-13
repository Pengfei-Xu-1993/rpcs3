if(NOT WIN32)
    message(FATAL_ERROR "The experimental RPCS3 DLSS integration currently supports Windows only.")
endif()

if(NOT USE_VULKAN)
    message(FATAL_ERROR "USE_NVIDIA_DLSS requires the Vulkan renderer (USE_VULKAN=ON).")
endif()

set(NVIDIA_DLSS_SDK_ROOT "" CACHE PATH "Path to the NVIDIA DLSS SDK root")

if(NOT NVIDIA_DLSS_SDK_ROOT)
    message(FATAL_ERROR "USE_NVIDIA_DLSS requires -DNVIDIA_DLSS_SDK_ROOT=<path-to-NVIDIA-DLSS-SDK>.")
endif()

set(_nvidia_dlss_include "${NVIDIA_DLSS_SDK_ROOT}/include")
set(_nvidia_dlss_library_paths
    "${NVIDIA_DLSS_SDK_ROOT}/lib/Windows_x86_64/x64"
    "${NVIDIA_DLSS_SDK_ROOT}/lib/Windows_x86_64/x86_64"
)

find_path(NVIDIA_DLSS_INCLUDE_DIR
    NAMES nvsdk_ngx_vk.h nvsdk_ngx_helpers_vk.h
    PATHS "${_nvidia_dlss_include}"
    NO_DEFAULT_PATH
)
find_library(NVIDIA_DLSS_NGX_RELEASE_LIBRARY
    NAMES nvsdk_ngx_d
    PATHS ${_nvidia_dlss_library_paths}
    NO_DEFAULT_PATH
)
find_library(NVIDIA_DLSS_NGX_DEBUG_LIBRARY
    NAMES nvsdk_ngx_d_dbg
    PATHS ${_nvidia_dlss_library_paths}
    NO_DEFAULT_PATH
)
find_file(NVIDIA_DLSS_RUNTIME_RELEASE
    NAMES nvngx_dlss.dll
    PATHS "${NVIDIA_DLSS_SDK_ROOT}/lib/Windows_x86_64/rel"
    NO_DEFAULT_PATH
)
find_file(NVIDIA_DLSS_RUNTIME_DEBUG
    NAMES nvngx_dlss.dll
    PATHS "${NVIDIA_DLSS_SDK_ROOT}/lib/Windows_x86_64/dev"
    NO_DEFAULT_PATH
)

if(NOT NVIDIA_DLSS_INCLUDE_DIR OR
   NOT EXISTS "${NVIDIA_DLSS_INCLUDE_DIR}/nvsdk_ngx_vk.h" OR
   NOT EXISTS "${NVIDIA_DLSS_INCLUDE_DIR}/nvsdk_ngx_helpers_vk.h" OR
   NOT NVIDIA_DLSS_NGX_RELEASE_LIBRARY OR
   NOT NVIDIA_DLSS_NGX_DEBUG_LIBRARY OR NOT NVIDIA_DLSS_RUNTIME_RELEASE OR
   NOT NVIDIA_DLSS_RUNTIME_DEBUG)
    message(FATAL_ERROR
        "The NVIDIA DLSS SDK at '${NVIDIA_DLSS_SDK_ROOT}' is incomplete. "
        "Expected include/nvsdk_ngx_vk.h, NGX debug/release libraries, and nvngx_dlss.dll."
    )
endif()

add_library(3rdparty_nvidia_dlss STATIC IMPORTED GLOBAL)
add_library(3rdparty::nvidia_dlss ALIAS 3rdparty_nvidia_dlss)

set_target_properties(3rdparty_nvidia_dlss PROPERTIES
    IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
    IMPORTED_LOCATION_DEBUG "${NVIDIA_DLSS_NGX_DEBUG_LIBRARY}"
    IMPORTED_LOCATION_RELEASE "${NVIDIA_DLSS_NGX_RELEASE_LIBRARY}"
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE
    MAP_IMPORTED_CONFIG_MINSIZEREL RELEASE
    INTERFACE_INCLUDE_DIRECTORIES "${NVIDIA_DLSS_INCLUDE_DIR}"
    INTERFACE_COMPILE_DEFINITIONS "RPCS3_HAS_NVIDIA_DLSS=1"
)

message(STATUS "NVIDIA DLSS SDK: ${NVIDIA_DLSS_SDK_ROOT}")
