//! Host memory availability for independently admitted embedding work.

/// Bytes the host still has free, or `None` when it cannot be determined.
///
/// Free + inactive + speculative: inactive and speculative pages are reclaimable on demand, so
/// counting only `free` would understate what is available by most of the file cache and refuse
/// work that would have fit.
///
/// **The page size is read, never assumed.** Apple Silicon uses 16 KiB pages where x86 used 4 KiB,
/// and hardcoding 4096 here understates availability by exactly 4x — which is precisely the error
/// that produced a wrong headline number while this was being characterised.
///
/// `None` on any failure. The caller treats that as "no opinion" and admits as it would without a
/// probe: a memory check that cannot read memory must not become a new way to refuse work.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(unsafe_code)]
pub fn host_available_bytes() -> Option<usize> {
    unsafe extern "C" {
        static mach_task_self_: libc::mach_port_t;
        fn mach_host_self() -> libc::mach_port_t;
        fn mach_port_deallocate(task: libc::mach_port_t, name: libc::mach_port_t) -> i32;
    }
    // SAFETY: the kernel writes at most `count` integer words into this initialized
    // ABI-sized structure. Each acquired host send right is released before returning.
    let (status, count, stats, page_size) = unsafe {
        let mut stats: libc::vm_statistics64 = std::mem::zeroed();
        let mut count = libc::HOST_VM_INFO64_COUNT;
        let host = mach_host_self();
        let status = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&mut stats as *mut libc::vm_statistics64).cast(),
            &mut count,
        );
        mach_port_deallocate(mach_task_self_, host);
        (status, count, stats, libc::vm_page_size)
    };
    // Mach's free_count already includes speculative pages; vm_stat subtracts them
    // from its "Pages free" display. Require the three leading counters we use.
    (status == libc::KERN_SUCCESS && count >= 3 && page_size > 0).then(|| {
        (stats.free_count as usize)
            .saturating_add(stats.inactive_count as usize)
            .saturating_mul(page_size)
    })
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn host_available_bytes() -> Option<usize> {
    None
}

#[cfg(test)]
mod host_memory_tests {
    #[test]
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn the_host_probe_reads_a_believable_number_from_the_real_machine() {
        // This exists because the bug it guards against already happened: the page size was assumed
        // to be 4096 while Apple Silicon uses 16384, understating availability by exactly 4x and
        // producing a wrong headline figure. An assumed-too-LARGE page size is detectable — it makes
        // "available" exceed physical RAM — so bound it against the real hardware size rather than
        // against a constant.
        let physical = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .and_then(|text| text.trim().parse::<usize>().ok())
            .expect("hw.memsize");
        let available = super::host_available_bytes().expect("probe should read this machine");
        assert!(
            available > 0 && available <= physical,
            "probe reported {available} bytes available on a machine with {physical} bytes of RAM"
        );
        // Not an assertion — the value itself, so a human can compare it against `vm_stat` when this
        // is being trusted for the first time on new hardware.
        println!(
            "host_available_bytes = {available} ({:.1} GB) of {:.1} GB physical",
            available as f64 / 2f64.powi(30),
            physical as f64 / 2f64.powi(30)
        );
    }
}
