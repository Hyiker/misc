add_rules("mode.debug", "mode.release")

target("crtp")
    set_kind("binary")
    add_files("crtp.cpp")
    set_languages("cxx17")
    add_defines("PRECISION_32", "_CRT_SECURE_NO_WARNINGS")