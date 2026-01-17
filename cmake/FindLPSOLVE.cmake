include(FindPackageHandleStandardArgs)

if(NOT LPSOLVE_ROOT)
  set(LPSOLVE_ROOT $ENV{LPSOLVE_ROOT})
endif()

find_path(LPSOLVE_INCLUDE_DIR NAMES lp_lib.h
  PATH_SUFFIXES lpsolve include
  ${_LPSOLVE_INC_SEARCH_OPTS}
  )

if(NOT LPSOLVE_LIBRARIES)
  # Fix debian nonsense:
  # see https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=503314
  if(EXISTS "/etc/debian_version")
    find_library(LPSOLVE_LIBRARIES NAMES liblpsolve55.so
      ${_LPSOLVE_SEARCH_OPTS}
      PATH_SUFFIXES lib/lp_solve lib)
  else()
    find_library(LPSOLVE_LIBRARIES NAMES lpsolve55
      ${_LPSOLVE_SEARCH_OPTS}
      PATH_SUFFIXES lib lib64)
  endif()
endif()

find_package_handle_standard_args(LPSOLVE
  REQUIRED_VARS LPSOLVE_LIBRARIES LPSOLVE_INCLUDE_DIR)

if(LPSOLVE_FOUND)
  
  if(NOT TARGET LPSOLVE::LPSOLVE)
    add_library(LPSOLVE::LPSOLVE IMPORTED INTERFACE)
    set_property(TARGET LPSOLVE::LPSOLVE PROPERTY INTERFACE_LINK_LIBRARIES ${LPSOLVE_LIBRARIES})
    if(LPSOLVE_INCLUDE_DIR)
      set_target_properties(LPSOLVE::LPSOLVE PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LPSOLVE_INCLUDE_DIR}")
    endif()
  endif()
endif()


