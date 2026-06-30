// Verilator C++ testbench for v1 MLP per M57 spec §6.6.
//
// Protocol:
//   - Reads 784 input bytes from stdin (the i8 activation vector).
//   - Applies reset for one clock cycle.
//   - Runs 3 clock cycles to propagate through both registered layer
//     boundaries (spec §6.6 cycle table).
//   - Prints each tap port to stdout as "<name>=0x<hex>" one per line.
//
// NOTE: This file is committed as infrastructure for M57.1 (v1 closure).
// It is NOT exercised on this branch — the v1 MLP Verilog is not yet
// synthesizable end-to-end (missing AST->KIR dispatch + HIR port/wire gen).
// See crates/nsl-test/tests/fpga_mlp_v1_parity.rs for the #[ignore]'d gate.
//
// Port packing notes:
//   Verilator splits wide ports into arrays of uint32_t or uint64_t.
//   The exact element types depend on the emitted .v port widths.
//   At M57.1 time, adjust the pack/unpack code below to match the actual
//   Verilator-generated Vtiny_mlp.h header. Specifically:
//     - x        : 784 × i8 → 6272-bit port → dut->x[k] (uint32_t array)
//     - tap_l1_*: 128 × i32 → 4096-bit port
//     - tap_l2_*: 10  × i64 → 640-bit port
//     - out      : 10  × i64 → 640-bit port

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <cstring>
#include <verilated.h>
#include "Vtiny_mlp.h"

constexpr int N_CYCLES = 3;  // worst-case across register-placement choices

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vtiny_mlp* dut = new Vtiny_mlp;

    // -----------------------------------------------------------------------
    // Read 784 i8 input values from stdin
    // -----------------------------------------------------------------------
    std::vector<int8_t> input(784);
    if (!std::cin.read(reinterpret_cast<char*>(input.data()), 784)) {
        std::cerr << "testbench: failed to read 784 input bytes\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // Pack input into wide port.
    // Verilator lays out a 6272-bit port as dut->x[k] where each element
    // is a uint32_t (Verilator VL_SIZEBITS=32 default) holding 4 packed i8
    // values.  Adjust if Verilator emits 64-bit elements for this target.
    // Layout: x[0] holds input[0..3], x[1] holds input[4..7], ...
    // -----------------------------------------------------------------------
    constexpr int X_WORDS = (784 + 3) / 4;  // 196 uint32_t words
    for (int i = 0; i < X_WORDS; i++) {
        uint32_t slot = 0;
        for (int b = 0; b < 4; b++) {
            int idx = i * 4 + b;
            if (idx < 784) {
                slot |= static_cast<uint32_t>(static_cast<uint8_t>(input[idx])) << (b * 8);
            }
        }
        dut->x[i] = slot;
    }

    // -----------------------------------------------------------------------
    // Apply synchronous reset for one cycle, then de-assert
    // -----------------------------------------------------------------------
    dut->clk = 0;
    dut->rst = 1;
    dut->eval();
    dut->clk = 1;
    dut->eval();
    dut->clk = 0;
    dut->rst = 0;
    dut->eval();

    // -----------------------------------------------------------------------
    // Run N_CYCLES = 3 cycles
    // -----------------------------------------------------------------------
    for (int c = 0; c < N_CYCLES; c++) {
        dut->clk = 1;
        dut->eval();
        dut->clk = 0;
        dut->eval();
    }

    // -----------------------------------------------------------------------
    // Dump tap ports: "<name>=0x<hex>\n"
    // Wide ports are arrays; we print MSB-first (highest index first) for
    // consistent big-endian hex regardless of Verilator element width.
    // -----------------------------------------------------------------------

    // Lambda: dump a uint32_t[] port with n_words elements.
    auto dump32 = [](const char* name, const uint32_t* port, int n_words) {
        std::cout << name << "=0x";
        for (int i = n_words - 1; i >= 0; i--) {
            std::cout << std::hex << std::setw(8) << std::setfill('0') << port[i];
        }
        std::cout << std::dec << "\n";
    };

    // Layer 1: 128 × i32 = 4096 bits = 128 uint32_t words.
    // Note: tap_l1_bias_out / tap_l2_bias_out were removed by M57.1 §3.5
    // bias-as-seed fold (wire-array mini §5) — bias is now folded into the
    // MAC accumulator seed, so the post-MAC bias-add wire no longer exists
    // in the generated DUT. matmul / relu / out taps remain.
    dump32("tap_l1_matmul_out", dut->tap_l1_matmul_out, 128);
    dump32("tap_l1_relu_out",   dut->tap_l1_relu_out,   128);

    // Layer 2: 10 × i64 = 640 bits = 20 uint32_t words
    dump32("tap_l2_matmul_out", dut->tap_l2_matmul_out, 20);
    dump32("tap_l2_relu_out",   dut->tap_l2_relu_out,   20);

    // Final output: 10 × i64 = 640 bits = 20 uint32_t words
    dump32("out", dut->out, 20);

    delete dut;
    return 0;
}
