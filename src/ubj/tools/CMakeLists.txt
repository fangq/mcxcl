add_definitions(-DUBJW_DISASSEMBLY_MODE)
add_executable(ubjdump ubjdump.c ../ubjr.c ../ubjrw.c ../ubjw.c)
target_link_libraries(ubjdump)

add_executable(ubjopt ubjopt.c)
target_link_libraries(ubjopt ubj)

