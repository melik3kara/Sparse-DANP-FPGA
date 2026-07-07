# Phase-1 portable CPU reference build.
# Plain C++17, no HLS, no external libraries.

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Iinclude
LDFLAGS  ?=

BIN      := test_phase1
SRC      := src/anp_reference.cpp test/test_phase1.cpp

# Default golden-vector directory (relative to this Makefile's directory).
GOLDEN   ?= ../phase0/golden_debug_16_8_4_2/bin

.PHONY: all run clean

all: $(BIN)

$(BIN): $(SRC) include/anp_config.hpp include/anp_types.hpp include/anp_reference.hpp
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

run: $(BIN)
	./$(BIN) $(GOLDEN)

clean:
	rm -f $(BIN)
