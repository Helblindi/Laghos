set(MFEM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../mfem/build/install
    CACHE PATH "absolute path to the MFEM build or install prefix")
set(mfem_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../mfem
    CACHE PATH "absolute path to where MFEMConfig.cmake is")

set(LAGLOS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Laglos/build/install
    CACHE PATH "absolute path to the LAGLOS build or install prefix")
set(laglos_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Laglos/build/install/lib/cmake
    CACHE PATH "absolute path to where LAGLOSConfig.cmake is")

# option(LAGLOS_USE_HIOP "Utilize sparse linear algebra" ON)
option(LAGHOS_LIMIT "Utilize low order Lagrangian solution to provide IDP baseline" ON)
