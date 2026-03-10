#[derive(Clone, Hash)]
/// Flags group `x86`.
pub struct Flags {
    bytes: [u8; 5],
}
impl Flags {
    /// Create flags x86 settings group.
    #[allow(unused_variables)]
    pub fn new(shared: &settings::Flags, builder: &Builder) -> Self {
        let bvec = builder.state_for("x86");
        let mut x86 = Self { bytes: [0; 5] };
        debug_assert_eq!(bvec.len(), 3);
        x86.bytes[0..3].copy_from_slice(&bvec);
        // Precompute #17.
        if x86.has_avx() {
            x86.bytes[2] |= 1 << 1;
        }
        // Precompute #18.
        if x86.has_avx() && x86.has_avx2() {
            x86.bytes[2] |= 1 << 2;
        }
        // Precompute #19.
        if x86.has_avx512bitalg() {
            x86.bytes[2] |= 1 << 3;
        }
        // Precompute #20.
        if x86.has_avx512dq() {
            x86.bytes[2] |= 1 << 4;
        }
        // Precompute #21.
        if x86.has_avx512f() {
            x86.bytes[2] |= 1 << 5;
        }
        // Precompute #22.
        if x86.has_avx512vbmi() {
            x86.bytes[2] |= 1 << 6;
        }
        // Precompute #23.
        if x86.has_avx512vl() {
            x86.bytes[2] |= 1 << 7;
        }
        // Precompute #24.
        if x86.has_bmi1() {
            x86.bytes[3] |= 1 << 0;
        }
        // Precompute #25.
        if x86.has_bmi2() {
            x86.bytes[3] |= 1 << 1;
        }
        // Precompute #26.
        if x86.has_cmpxchg16b() {
            x86.bytes[3] |= 1 << 2;
        }
        // Precompute #27.
        if x86.has_avx() && x86.has_fma() {
            x86.bytes[3] |= 1 << 3;
        }
        // Precompute #28.
        if x86.has_lzcnt() {
            x86.bytes[3] |= 1 << 4;
        }
        // Precompute #29.
        if x86.has_popcnt() && x86.has_sse42() {
            x86.bytes[3] |= 1 << 5;
        }
        // Precompute #30.
        if x86.has_sse41() {
            x86.bytes[3] |= 1 << 6;
        }
        // Precompute #31.
        if x86.has_sse41() && x86.has_sse42() {
            x86.bytes[3] |= 1 << 7;
        }
        // Precompute #32.
        if x86.has_ssse3() {
            x86.bytes[4] |= 1 << 0;
        }
        x86
    }
}
impl Flags {
    /// Iterates the setting values.
    pub fn iter(&self) -> impl Iterator<Item = Value> {
        let mut bytes = [0; 3];
        bytes.copy_from_slice(&self.bytes[0..3]);
        DESCRIPTORS.iter().filter_map(move |d| {
            let values = match &d.detail {
                detail::Detail::Preset => return None,
                detail::Detail::Enum { last, enumerators } => Some(TEMPLATE.enums(*last, *enumerators)),
                _ => None
            };
            Some(Value{ name: d.name, detail: d.detail, values, value: bytes[d.offset as usize] })
        })
    }
}
/// User-defined settings.
#[allow(dead_code)]
impl Flags {
    /// Get a view of the boolean predicates.
    pub fn predicate_view(&self) -> crate::settings::PredicateView {
        crate::settings::PredicateView::new(&self.bytes[0..])
    }
    /// Dynamic numbered predicate getter.
    fn numbered_predicate(&self, p: usize) -> bool {
        self.bytes[0 + p / 8] & (1 << (p % 8)) != 0
    }
    /// Has support for SSE3.
    /// SSE3: CPUID.01H:ECX.SSE3[bit 0]
    pub fn has_sse3(&self) -> bool {
        self.numbered_predicate(0)
    }
    /// Has support for SSSE3.
    /// SSSE3: CPUID.01H:ECX.SSSE3[bit 9]
    pub fn has_ssse3(&self) -> bool {
        self.numbered_predicate(1)
    }
    /// Has support for CMPXCHG16b.
    /// CMPXCHG16b: CPUID.01H:ECX.CMPXCHG16B[bit 13]
    pub fn has_cmpxchg16b(&self) -> bool {
        self.numbered_predicate(2)
    }
    /// Has support for SSE4.1.
    /// SSE4.1: CPUID.01H:ECX.SSE4_1[bit 19]
    pub fn has_sse41(&self) -> bool {
        self.numbered_predicate(3)
    }
    /// Has support for SSE4.2.
    /// SSE4.2: CPUID.01H:ECX.SSE4_2[bit 20]
    pub fn has_sse42(&self) -> bool {
        self.numbered_predicate(4)
    }
    /// Has support for AVX.
    /// AVX: CPUID.01H:ECX.AVX[bit 28]
    pub fn has_avx(&self) -> bool {
        self.numbered_predicate(5)
    }
    /// Has support for AVX2.
    /// AVX2: CPUID.07H:EBX.AVX2[bit 5]
    pub fn has_avx2(&self) -> bool {
        self.numbered_predicate(6)
    }
    /// Has support for FMA.
    /// FMA: CPUID.01H:ECX.FMA[bit 12]
    pub fn has_fma(&self) -> bool {
        self.numbered_predicate(7)
    }
    /// Has support for AVX512BITALG.
    /// AVX512BITALG: CPUID.07H:ECX.AVX512BITALG[bit 12]
    pub fn has_avx512bitalg(&self) -> bool {
        self.numbered_predicate(8)
    }
    /// Has support for AVX512DQ.
    /// AVX512DQ: CPUID.07H:EBX.AVX512DQ[bit 17]
    pub fn has_avx512dq(&self) -> bool {
        self.numbered_predicate(9)
    }
    /// Has support for AVX512VL.
    /// AVX512VL: CPUID.07H:EBX.AVX512VL[bit 31]
    pub fn has_avx512vl(&self) -> bool {
        self.numbered_predicate(10)
    }
    /// Has support for AVX512VMBI.
    /// AVX512VBMI: CPUID.07H:ECX.AVX512VBMI[bit 1]
    pub fn has_avx512vbmi(&self) -> bool {
        self.numbered_predicate(11)
    }
    /// Has support for AVX512F.
    /// AVX512F: CPUID.07H:EBX.AVX512F[bit 16]
    pub fn has_avx512f(&self) -> bool {
        self.numbered_predicate(12)
    }
    /// Has support for POPCNT.
    /// POPCNT: CPUID.01H:ECX.POPCNT[bit 23]
    pub fn has_popcnt(&self) -> bool {
        self.numbered_predicate(13)
    }
    /// Has support for BMI1.
    /// BMI1: CPUID.(EAX=07H, ECX=0H):EBX.BMI1[bit 3]
    pub fn has_bmi1(&self) -> bool {
        self.numbered_predicate(14)
    }
    /// Has support for BMI2.
    /// BMI2: CPUID.(EAX=07H, ECX=0H):EBX.BMI2[bit 8]
    pub fn has_bmi2(&self) -> bool {
        self.numbered_predicate(15)
    }
    /// Has support for LZCNT.
    /// LZCNT: CPUID.EAX=80000001H:ECX.LZCNT[bit 5]
    pub fn has_lzcnt(&self) -> bool {
        self.numbered_predicate(16)
    }
    /// Computed predicate `x86.has_avx()`.
    pub fn use_avx(&self) -> bool {
        self.numbered_predicate(17)
    }
    /// Computed predicate `x86.has_avx() && x86.has_avx2()`.
    pub fn use_avx2(&self) -> bool {
        self.numbered_predicate(18)
    }
    /// Computed predicate `x86.has_avx512bitalg()`.
    pub fn use_avx512bitalg(&self) -> bool {
        self.numbered_predicate(19)
    }
    /// Computed predicate `x86.has_avx512dq()`.
    pub fn use_avx512dq(&self) -> bool {
        self.numbered_predicate(20)
    }
    /// Computed predicate `x86.has_avx512f()`.
    pub fn use_avx512f(&self) -> bool {
        self.numbered_predicate(21)
    }
    /// Computed predicate `x86.has_avx512vbmi()`.
    pub fn use_avx512vbmi(&self) -> bool {
        self.numbered_predicate(22)
    }
    /// Computed predicate `x86.has_avx512vl()`.
    pub fn use_avx512vl(&self) -> bool {
        self.numbered_predicate(23)
    }
    /// Computed predicate `x86.has_bmi1()`.
    pub fn use_bmi1(&self) -> bool {
        self.numbered_predicate(24)
    }
    /// Computed predicate `x86.has_bmi2()`.
    pub fn use_bmi2(&self) -> bool {
        self.numbered_predicate(25)
    }
    /// Computed predicate `x86.has_cmpxchg16b()`.
    pub fn use_cmpxchg16b(&self) -> bool {
        self.numbered_predicate(26)
    }
    /// Computed predicate `x86.has_avx() && x86.has_fma()`.
    pub fn use_fma(&self) -> bool {
        self.numbered_predicate(27)
    }
    /// Computed predicate `x86.has_lzcnt()`.
    pub fn use_lzcnt(&self) -> bool {
        self.numbered_predicate(28)
    }
    /// Computed predicate `x86.has_popcnt() && x86.has_sse42()`.
    pub fn use_popcnt(&self) -> bool {
        self.numbered_predicate(29)
    }
    /// Computed predicate `x86.has_sse41()`.
    pub fn use_sse41(&self) -> bool {
        self.numbered_predicate(30)
    }
    /// Computed predicate `x86.has_sse41() && x86.has_sse42()`.
    pub fn use_sse42(&self) -> bool {
        self.numbered_predicate(31)
    }
    /// Computed predicate `x86.has_ssse3()`.
    pub fn use_ssse3(&self) -> bool {
        self.numbered_predicate(32)
    }
}
static DESCRIPTORS: [detail::Descriptor; 84] = [
    detail::Descriptor {
        name: "has_sse3",
        description: "Has support for SSE3.",
        offset: 0,
        detail: detail::Detail::Bool { bit: 0 },
    },
    detail::Descriptor {
        name: "has_ssse3",
        description: "Has support for SSSE3.",
        offset: 0,
        detail: detail::Detail::Bool { bit: 1 },
    },
    detail::Descriptor {
        name: "has_cmpxchg16b",
        description: "Has support for CMPXCHG16b.",
        offset: 0,
        detail: detail::Detail::Bool { bit: 2 },
    },
    detail::Descriptor {
        name: "has_sse41",
        description: "Has support for SSE4.1.",
        offset: 0,
        detail: detail::Detail::Bool { bit: 3 },
    },
    detail::Descriptor {
        name: "has_sse42",
        description: "Has support for SSE4.2.",
        offset: 0,
        detail: detail::Detail::Bool { bit: 4 },
    },
    detail::Descriptor {
        name: "has_avx",
        description: "Has support for AVX.",
        offset: 0,
        detail: detail::Detail::Bool { bit: 5 },
    },
    detail::Descriptor {
        name: "has_avx2",
        description: "Has support for AVX2.",
        offset: 0,
        detail: detail::Detail::Bool { bit: 6 },
    },
    detail::Descriptor {
        name: "has_fma",
        description: "Has support for FMA.",
        offset: 0,
        detail: detail::Detail::Bool { bit: 7 },
    },
    detail::Descriptor {
        name: "has_avx512bitalg",
        description: "Has support for AVX512BITALG.",
        offset: 1,
        detail: detail::Detail::Bool { bit: 0 },
    },
    detail::Descriptor {
        name: "has_avx512dq",
        description: "Has support for AVX512DQ.",
        offset: 1,
        detail: detail::Detail::Bool { bit: 1 },
    },
    detail::Descriptor {
        name: "has_avx512vl",
        description: "Has support for AVX512VL.",
        offset: 1,
        detail: detail::Detail::Bool { bit: 2 },
    },
    detail::Descriptor {
        name: "has_avx512vbmi",
        description: "Has support for AVX512VMBI.",
        offset: 1,
        detail: detail::Detail::Bool { bit: 3 },
    },
    detail::Descriptor {
        name: "has_avx512f",
        description: "Has support for AVX512F.",
        offset: 1,
        detail: detail::Detail::Bool { bit: 4 },
    },
    detail::Descriptor {
        name: "has_popcnt",
        description: "Has support for POPCNT.",
        offset: 1,
        detail: detail::Detail::Bool { bit: 5 },
    },
    detail::Descriptor {
        name: "has_bmi1",
        description: "Has support for BMI1.",
        offset: 1,
        detail: detail::Detail::Bool { bit: 6 },
    },
    detail::Descriptor {
        name: "has_bmi2",
        description: "Has support for BMI2.",
        offset: 1,
        detail: detail::Detail::Bool { bit: 7 },
    },
    detail::Descriptor {
        name: "has_lzcnt",
        description: "Has support for LZCNT.",
        offset: 2,
        detail: detail::Detail::Bool { bit: 0 },
    },
    detail::Descriptor {
        name: "sse3",
        description: "SSE3 and earlier.",
        offset: 0,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "ssse3",
        description: "SSSE3 and earlier.",
        offset: 3,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "sse41",
        description: "SSE4.1 and earlier.",
        offset: 6,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "sse42",
        description: "SSE4.2 and earlier.",
        offset: 9,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "baseline",
        description: "A baseline preset with no extensions enabled.",
        offset: 12,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "nocona",
        description: "Nocona microarchitecture.",
        offset: 15,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "core2",
        description: "Core 2 microarchitecture.",
        offset: 18,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "penryn",
        description: "Penryn microarchitecture.",
        offset: 21,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "atom",
        description: "Atom microarchitecture.",
        offset: 24,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "bonnell",
        description: "Bonnell microarchitecture.",
        offset: 27,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "silvermont",
        description: "Silvermont microarchitecture.",
        offset: 30,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "slm",
        description: "Silvermont microarchitecture.",
        offset: 33,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "goldmont",
        description: "Goldmont microarchitecture.",
        offset: 36,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "goldmont-plus",
        description: "Goldmont Plus microarchitecture.",
        offset: 39,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "tremont",
        description: "Tremont microarchitecture.",
        offset: 42,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "alderlake",
        description: "Alderlake microarchitecture.",
        offset: 45,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "sierraforest",
        description: "Sierra Forest microarchitecture.",
        offset: 48,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "grandridge",
        description: "Grandridge microarchitecture.",
        offset: 51,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "nehalem",
        description: "Nehalem microarchitecture.",
        offset: 54,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "corei7",
        description: "Core i7 microarchitecture.",
        offset: 57,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "westmere",
        description: "Westmere microarchitecture.",
        offset: 60,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "sandybridge",
        description: "Sandy Bridge microarchitecture.",
        offset: 63,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "corei7-avx",
        description: "Core i7 AVX microarchitecture.",
        offset: 66,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "ivybridge",
        description: "Ivy Bridge microarchitecture.",
        offset: 69,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "core-avx-i",
        description: "Intel Core CPU with 64-bit extensions.",
        offset: 72,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "haswell",
        description: "Haswell microarchitecture.",
        offset: 75,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "core-avx2",
        description: "Intel Core CPU with AVX2 extensions.",
        offset: 78,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "broadwell",
        description: "Broadwell microarchitecture.",
        offset: 81,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "skylake",
        description: "Skylake microarchitecture.",
        offset: 84,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "knl",
        description: "Knights Landing microarchitecture.",
        offset: 87,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "knm",
        description: "Knights Mill microarchitecture.",
        offset: 90,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "skylake-avx512",
        description: "Skylake AVX512 microarchitecture.",
        offset: 93,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "skx",
        description: "Skylake AVX512 microarchitecture.",
        offset: 96,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "cascadelake",
        description: "Cascade Lake microarchitecture.",
        offset: 99,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "cooperlake",
        description: "Cooper Lake microarchitecture.",
        offset: 102,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "cannonlake",
        description: "Canon Lake microarchitecture.",
        offset: 105,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "icelake-client",
        description: "Ice Lake microarchitecture.",
        offset: 108,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "icelake",
        description: "Ice Lake microarchitecture",
        offset: 111,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "icelake-server",
        description: "Ice Lake (server) microarchitecture.",
        offset: 114,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "tigerlake",
        description: "Tiger Lake microarchitecture.",
        offset: 117,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "sapphirerapids",
        description: "Sapphire Rapids microarchitecture.",
        offset: 120,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "raptorlake",
        description: "Raptor Lake microarchitecture.",
        offset: 123,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "meteorlake",
        description: "Meteor Lake microarchitecture.",
        offset: 126,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "graniterapids",
        description: "Granite Rapids microarchitecture.",
        offset: 129,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "opteron",
        description: "Opteron microarchitecture.",
        offset: 132,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "k8",
        description: "K8 Hammer microarchitecture.",
        offset: 135,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "athlon64",
        description: "Athlon64 microarchitecture.",
        offset: 138,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "athlon-fx",
        description: "Athlon FX microarchitecture.",
        offset: 141,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "opteron-sse3",
        description: "Opteron microarchitecture with support for SSE3 instructions.",
        offset: 144,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "k8-sse3",
        description: "K8 Hammer microarchitecture with support for SSE3 instructions.",
        offset: 147,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "athlon64-sse3",
        description: "Athlon 64 microarchitecture with support for SSE3 instructions.",
        offset: 150,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "barcelona",
        description: "Barcelona microarchitecture.",
        offset: 153,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "amdfam10",
        description: "AMD Family 10h microarchitecture",
        offset: 156,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "btver1",
        description: "Bobcat microarchitecture.",
        offset: 159,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "btver2",
        description: "Jaguar microarchitecture.",
        offset: 162,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "bdver1",
        description: "Bulldozer microarchitecture",
        offset: 165,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "bdver2",
        description: "Piledriver microarchitecture.",
        offset: 168,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "bdver3",
        description: "Steamroller microarchitecture.",
        offset: 171,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "bdver4",
        description: "Excavator microarchitecture.",
        offset: 174,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "znver1",
        description: "Zen (first generation) microarchitecture.",
        offset: 177,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "znver2",
        description: "Zen (second generation) microarchitecture.",
        offset: 180,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "znver3",
        description: "Zen (third generation) microarchitecture.",
        offset: 183,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "znver4",
        description: "Zen (fourth generation) microarchitecture.",
        offset: 186,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "x86-64",
        description: "Generic x86-64 microarchitecture.",
        offset: 189,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "x86-64-v2",
        description: "Generic x86-64 (V2) microarchitecture.",
        offset: 192,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "x84_64_v3",
        description: "Generic x86_64 (V3) microarchitecture.",
        offset: 195,
        detail: detail::Detail::Preset,
    },
    detail::Descriptor {
        name: "x86_64_v4",
        description: "Generic x86_64 (V4) microarchitecture.",
        offset: 198,
        detail: detail::Detail::Preset,
    },
];
static ENUMERATORS: [&str; 0] = [
];
static HASH_TABLE: [u16; 128] = [
    0xffff,
    0xffff,
    78,
    77,
    76,
    0xffff,
    0xffff,
    0xffff,
    24,
    79,
    67,
    81,
    23,
    51,
    60,
    15,
    14,
    30,
    1,
    42,
    71,
    68,
    5,
    36,
    0xffff,
    66,
    6,
    45,
    22,
    65,
    16,
    7,
    48,
    50,
    25,
    63,
    0xffff,
    12,
    44,
    39,
    53,
    0xffff,
    0xffff,
    70,
    0xffff,
    4,
    32,
    0xffff,
    3,
    0xffff,
    0xffff,
    59,
    0xffff,
    0xffff,
    11,
    13,
    0xffff,
    0xffff,
    0xffff,
    0xffff,
    0xffff,
    0xffff,
    0xffff,
    0xffff,
    31,
    80,
    74,
    0,
    40,
    29,
    47,
    46,
    9,
    55,
    72,
    10,
    75,
    73,
    2,
    0xffff,
    0xffff,
    62,
    82,
    34,
    8,
    0xffff,
    19,
    20,
    49,
    17,
    54,
    61,
    0xffff,
    0xffff,
    21,
    0xffff,
    64,
    69,
    57,
    0xffff,
    0xffff,
    83,
    0xffff,
    27,
    28,
    0xffff,
    35,
    0xffff,
    0xffff,
    37,
    0xffff,
    0xffff,
    41,
    43,
    0xffff,
    33,
    0xffff,
    0xffff,
    0xffff,
    58,
    52,
    0xffff,
    0xffff,
    18,
    56,
    0xffff,
    26,
    38,
];
static PRESETS: [(u8, u8); 201] = [
    // sse3: has_sse3
    (0b00000001, 0b00000001),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // ssse3: has_sse3, has_ssse3
    (0b00000011, 0b00000011),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // sse41: has_sse3, has_ssse3, has_sse41
    (0b00001011, 0b00001011),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // sse42: has_sse3, has_ssse3, has_sse41, has_sse42
    (0b00011011, 0b00011011),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // baseline: 
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // nocona: has_sse3, has_cmpxchg16b
    (0b00000101, 0b00000101),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // core2: has_sse3, has_cmpxchg16b
    (0b00000101, 0b00000101),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // penryn: has_sse3, has_ssse3, has_sse41, has_cmpxchg16b
    (0b00001111, 0b00001111),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // atom: has_sse3, has_ssse3, has_cmpxchg16b
    (0b00000111, 0b00000111),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // bonnell: has_sse3, has_ssse3, has_cmpxchg16b
    (0b00000111, 0b00000111),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // silvermont: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt
    (0b00011111, 0b00011111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // slm: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt
    (0b00011111, 0b00011111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // goldmont: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt
    (0b00011111, 0b00011111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // goldmont-plus: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt
    (0b00011111, 0b00011111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // tremont: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt
    (0b00011111, 0b00011111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // alderlake: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma
    (0b10011111, 0b10011111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // sierraforest: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma
    (0b10011111, 0b10011111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // grandridge: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma
    (0b10011111, 0b10011111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // nehalem: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b
    (0b00011111, 0b00011111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // corei7: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b
    (0b00011111, 0b00011111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // westmere: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b
    (0b00011111, 0b00011111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // sandybridge: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx
    (0b00111111, 0b00111111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // corei7-avx: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx
    (0b00111111, 0b00111111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // ivybridge: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx
    (0b00111111, 0b00111111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // core-avx-i: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx
    (0b00111111, 0b00111111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // haswell: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt
    (0b11111111, 0b11111111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // core-avx2: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt
    (0b11111111, 0b11111111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // broadwell: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt
    (0b11111111, 0b11111111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // skylake: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt
    (0b11111111, 0b11111111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // knl: has_popcnt, has_avx512f, has_fma, has_bmi1, has_bmi2, has_lzcnt, has_cmpxchg16b
    (0b10000100, 0b10000100),
    (0b11110000, 0b11110000),
    (0b00000001, 0b00000001),
    // knm: has_popcnt, has_avx512f, has_fma, has_bmi1, has_bmi2, has_lzcnt, has_cmpxchg16b
    (0b10000100, 0b10000100),
    (0b11110000, 0b11110000),
    (0b00000001, 0b00000001),
    // skylake-avx512: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl
    (0b11111111, 0b11111111),
    (0b11110110, 0b11110110),
    (0b00000001, 0b00000001),
    // skx: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl
    (0b11111111, 0b11111111),
    (0b11110110, 0b11110110),
    (0b00000001, 0b00000001),
    // cascadelake: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl
    (0b11111111, 0b11111111),
    (0b11110110, 0b11110110),
    (0b00000001, 0b00000001),
    // cooperlake: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl
    (0b11111111, 0b11111111),
    (0b11110110, 0b11110110),
    (0b00000001, 0b00000001),
    // cannonlake: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi
    (0b11111111, 0b11111111),
    (0b11111110, 0b11111110),
    (0b00000001, 0b00000001),
    // icelake-client: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi, has_avx512bitalg
    (0b11111111, 0b11111111),
    (0b11111111, 0b11111111),
    (0b00000001, 0b00000001),
    // icelake: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi, has_avx512bitalg
    (0b11111111, 0b11111111),
    (0b11111111, 0b11111111),
    (0b00000001, 0b00000001),
    // icelake-server: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi, has_avx512bitalg
    (0b11111111, 0b11111111),
    (0b11111111, 0b11111111),
    (0b00000001, 0b00000001),
    // tigerlake: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi, has_avx512bitalg
    (0b11111111, 0b11111111),
    (0b11111111, 0b11111111),
    (0b00000001, 0b00000001),
    // sapphirerapids: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi, has_avx512bitalg
    (0b11111111, 0b11111111),
    (0b11111111, 0b11111111),
    (0b00000001, 0b00000001),
    // raptorlake: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma
    (0b10011111, 0b10011111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // meteorlake: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma
    (0b10011111, 0b10011111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // graniterapids: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi, has_avx512bitalg
    (0b11111111, 0b11111111),
    (0b11111111, 0b11111111),
    (0b00000001, 0b00000001),
    // opteron: 
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // k8: 
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // athlon64: 
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // athlon-fx: 
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // opteron-sse3: has_sse3, has_cmpxchg16b
    (0b00000101, 0b00000101),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // k8-sse3: has_sse3, has_cmpxchg16b
    (0b00000101, 0b00000101),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // athlon64-sse3: has_sse3, has_cmpxchg16b
    (0b00000101, 0b00000101),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // barcelona: has_popcnt, has_lzcnt, has_cmpxchg16b
    (0b00000100, 0b00000100),
    (0b00100000, 0b00100000),
    (0b00000001, 0b00000001),
    // amdfam10: has_popcnt, has_lzcnt, has_cmpxchg16b
    (0b00000100, 0b00000100),
    (0b00100000, 0b00100000),
    (0b00000001, 0b00000001),
    // btver1: has_sse3, has_ssse3, has_lzcnt, has_popcnt, has_cmpxchg16b
    (0b00000111, 0b00000111),
    (0b00100000, 0b00100000),
    (0b00000001, 0b00000001),
    // btver2: has_sse3, has_ssse3, has_lzcnt, has_popcnt, has_cmpxchg16b, has_avx, has_bmi1
    (0b00100111, 0b00100111),
    (0b01100000, 0b01100000),
    (0b00000001, 0b00000001),
    // bdver1: has_lzcnt, has_popcnt, has_sse3, has_ssse3, has_cmpxchg16b
    (0b00000111, 0b00000111),
    (0b00100000, 0b00100000),
    (0b00000001, 0b00000001),
    // bdver2: has_lzcnt, has_popcnt, has_sse3, has_ssse3, has_cmpxchg16b, has_bmi1
    (0b00000111, 0b00000111),
    (0b01100000, 0b01100000),
    (0b00000001, 0b00000001),
    // bdver3: has_lzcnt, has_popcnt, has_sse3, has_ssse3, has_cmpxchg16b, has_bmi1
    (0b00000111, 0b00000111),
    (0b01100000, 0b01100000),
    (0b00000001, 0b00000001),
    // bdver4: has_lzcnt, has_popcnt, has_sse3, has_ssse3, has_cmpxchg16b, has_bmi1, has_avx2, has_bmi2
    (0b01000111, 0b01000111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // znver1: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma, has_cmpxchg16b
    (0b10011111, 0b10011111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // znver2: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma, has_cmpxchg16b
    (0b10011111, 0b10011111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // znver3: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma, has_cmpxchg16b
    (0b10011111, 0b10011111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // znver4: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma, has_cmpxchg16b, has_avx512bitalg, has_avx512dq, has_avx512f, has_avx512vbmi, has_avx512vl
    (0b10011111, 0b10011111),
    (0b11111111, 0b11111111),
    (0b00000001, 0b00000001),
    // x86-64: 
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    (0b00000000, 0b00000000),
    // x86-64-v2: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b
    (0b00011111, 0b00011111),
    (0b00100000, 0b00100000),
    (0b00000000, 0b00000000),
    // x84_64_v3: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx2
    (0b11011111, 0b11011111),
    (0b11100000, 0b11100000),
    (0b00000001, 0b00000001),
    // x86_64_v4: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx2, has_avx512dq, has_avx512vl
    (0b11011111, 0b11011111),
    (0b11100110, 0b11100110),
    (0b00000001, 0b00000001),
];
static TEMPLATE: detail::Template = detail::Template {
    name: "x86",
    descriptors: &DESCRIPTORS,
    enumerators: &ENUMERATORS,
    hash_table: &HASH_TABLE,
    defaults: &[0x00, 0x00, 0x00],
    presets: &PRESETS,
};
/// Create a `settings::Builder` for the x86 settings group.
pub fn builder() -> Builder {
    Builder::new(&TEMPLATE)
}
impl fmt::Display for Flags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[x86]")?;
        for d in &DESCRIPTORS {
            if !d.detail.is_preset() {
                write!(f, "{} = ", d.name)?;
                TEMPLATE.format_toml_value(d.detail, self.bytes[d.offset as usize], f)?;
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
