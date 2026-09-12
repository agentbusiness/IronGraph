//! Process-wide serialisation of each Candle Metal device's command stream.
//!
//! Candle 0.11's Metal backend keeps one buffer pool pair, one `MTLResidencySet`, and one
//! command-buffer chain per `MetalDevice`, and it mutates all three without a single covering
//! lock:
//!
//! * `MetalDevice::allocate_buffer` and `new_buffer_with_data` hold the *shared* pool lock while
//!   `new_buffer` holds the *private* pool lock, and all three then call
//!   `ResidencySet::insert` -> `addAllocation` + `commit` on the one residency set. A residency
//!   set is not thread-safe, so two allocations on two threads corrupt it.
//! * Every device-to-host readback runs `MetalDevice::flush_and_wait_current`, which waits only
//!   on the caller's own command buffer and then sweeps both pools in `drop_unused_buffers`,
//!   unregistering and freeing every buffer whose `Arc` count fell to one — including buffers
//!   another thread's still-in-flight command buffer references.
//!
//! Both are reachable the moment two threads share one `MetalDevice`, which is exactly what
//! `MetalBackend::pin_project` (device cloned per concurrent read) and the shared semantic
//! encoder device do. The observed failure is a SIGSEGV on `com.Metal.CompletionQueueDispatch`
//! inside `MTLResourceList releaseAllObjectsAndReset` while a worker thread sits in
//! `-[MTLResidencySet commit]`.
//!
//! The gate below makes one Candle device single-threaded again. It is reentrant, so a gated
//! backend method may call another gated method on the same thread, and it is keyed by Candle's
//! `DeviceId` so independent devices (graph residency vs. semantic encoder) still overlap.

use candle_core::Device;
use parking_lot::{ReentrantMutex, ReentrantMutexGuard};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::collections::HashMap;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::sync::{Mutex, OnceLock};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use candle_core::metal_backend::DeviceId;

#[cfg(any(target_os = "macos", target_os = "ios"))]
type GateRegistry = Mutex<HashMap<DeviceId, &'static ReentrantMutex<()>>>;

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn registry() -> &'static GateRegistry {
    static REGISTRY: OnceLock<GateRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The gate for one Candle Metal device, created on first use and never destroyed.
///
/// Leaking is deliberate: a process constructs a handful of Candle devices and the guards handed
/// out here are `'static`, which is what lets a gate be held across an ordinary call stack
/// without threading a lifetime through every backend signature.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn gate_for(id: DeviceId) -> &'static ReentrantMutex<()> {
    let mut registry = match registry().lock() {
        Ok(registry) => registry,
        Err(poisoned) => poisoned.into_inner(),
    };
    registry
        .entry(id)
        .or_insert_with(|| &*Box::leak(Box::new(ReentrantMutex::new(()))))
}

/// The serialisation gate for `device`, or `None` when it is not a Metal device.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn metal_device_gate(device: &Device) -> Option<&'static ReentrantMutex<()>> {
    match device {
        Device::Metal(metal) => Some(gate_for(metal.id())),
        _ => None,
    }
}

/// No Metal device exists off Apple platforms, so there is nothing to serialise.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn metal_device_gate(_device: &Device) -> Option<&'static ReentrantMutex<()>> {
    None
}

/// Holds `device` for the caller's thread. Non-Metal devices are not gated and return `None`.
///
/// The guard must cover every Candle operation on that device *including* the readback, because
/// the readback is what runs Candle's pool sweep.
#[must_use = "the device is only serialised while the guard is alive"]
pub fn lock_metal_device(device: &Device) -> Option<ReentrantMutexGuard<'static, ()>> {
    metal_device_gate(device).map(ReentrantMutex::lock)
}

/// Measures what the gate costs, because serialising a device is not free and a crash fix that
/// halves throughput trades a visible failure for an invisible one.
///
/// Run one arm per process. The `ungated` arm reproduces the pre-fix behaviour, which is
/// undefined behaviour by construction — it is allowed to take the test process down, and that
/// outcome is itself a result worth recording:
///
/// ```text
/// IRONGRAPH_METAL_GATE_BENCH=gated   cargo test -p irongraph-gpu --features accelerator \
///     metal_gate::throughput -- --ignored --nocapture
/// IRONGRAPH_METAL_GATE_BENCH=ungated cargo test -p irongraph-gpu --features accelerator \
///     metal_gate::throughput -- --ignored --nocapture
/// ```
#[cfg(all(test, target_os = "macos"))]
mod throughput {
    use std::time::Instant;

    use candle_core::{DType, Device, Tensor};

    /// One allocate -> compute -> read-back cycle, which is the shape of every device operation
    /// in the server: the read-back is what runs Candle's pool sweep, so it is the part the gate
    /// has to cover and therefore the part whose cost has to be measured.
    fn cycle(device: &Device, dim: usize) -> candle_core::Result<f32> {
        let left = Tensor::ones((dim, dim), DType::F32, device)?;
        let product = left.matmul(&left)?;
        product.sum_all()?.to_scalar::<f32>()
    }

    /// Cycles per second, and how many cycles actually succeeded.
    ///
    /// Failures are counted, not discarded: an arm that errors immediately would otherwise post a
    /// spectacular rate and read as "no regression".
    fn run_arm(
        device: &Device,
        dim: usize,
        workers: usize,
        iterations: usize,
        gated: bool,
    ) -> (f64, usize, usize) {
        let failures = std::sync::atomic::AtomicUsize::new(0);
        let started = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    for _ in 0..iterations {
                        let _gate = if gated {
                            super::lock_metal_device(device)
                        } else {
                            None
                        };
                        if cycle(device, dim).is_err() {
                            failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        let elapsed = started.elapsed().as_secs_f64();
        let attempted = workers * iterations;
        let failed = failures.load(std::sync::atomic::Ordering::Relaxed);
        let succeeded = attempted.saturating_sub(failed);
        if elapsed <= 0.0 {
            return (0.0, succeeded, failed);
        }
        (succeeded as f64 / elapsed, succeeded, failed)
    }

    #[test]
    #[ignore = "measures Metal device throughput on real hardware; one arm per process"]
    fn gate_throughput_against_ungated_baseline() {
        let _serialise = crate::metal_test_guard();
        let Some(device) = crate::metal_test_device() else {
            println!("no Metal device on this host; nothing measured");
            return;
        };
        let gated = match std::env::var("IRONGRAPH_METAL_GATE_BENCH").as_deref() {
            Ok("ungated") => false,
            _ => true,
        };
        // Warm the pipelines and the buffer pool so the first timed cycle is not paying for
        // shader compilation.
        for _ in 0..8 {
            let _ = cycle(&device, 128);
        }
        println!("arm={}", if gated { "gated" } else { "ungated" });
        for dim in [128_usize, 512] {
            for workers in [1_usize, 4, 8] {
                let iterations = if dim == 128 { 400 } else { 100 };
                let (rate, succeeded, failed) = run_arm(&device, dim, workers, iterations, gated);
                println!(
                    "dim={dim} workers={workers} succeeded={succeeded} failed={failed} \
                     cycles_per_second={rate:.1}"
                );
            }
        }
    }
}
