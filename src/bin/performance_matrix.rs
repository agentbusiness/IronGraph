// Keep the executable and custom-benchmark entrypoints on one implementation. `cargo run --bin
// performance-matrix` avoids compiling and LTO-linking the unrelated production server binary
// during large release measurements; the registered bench target remains available for CI smoke.
#[path = "../../benches/performance_matrix.rs"]
mod performance_matrix;

fn main() {
    performance_matrix::main();
}
