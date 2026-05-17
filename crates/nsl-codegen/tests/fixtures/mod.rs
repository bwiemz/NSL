// PCA Tier B fixture matrix for M3 parity per spec §4.4.
// Each fixture pins (seq_len, num_docs, doc_lengths, doc_offsets, padding_locs).

#[derive(Debug, Clone)]
pub struct PackingFixture {
    pub name: &'static str,
    pub seq_len: u32,
    pub doc_lengths: Vec<u32>,
    pub doc_offsets: Vec<u32>,
    pub padding_locs: Vec<u32>,
}

pub fn fixture_matrix() -> Vec<PackingFixture> {
    vec![
        PackingFixture {
            name: "standard_3doc",
            seq_len: 4096,
            doc_lengths: vec![1366, 1366, 1364],
            doc_offsets: vec![0, 1366, 2732],
            padding_locs: vec![],
        },
        PackingFixture {
            name: "long_seq_5doc",
            seq_len: 16_384,
            doc_lengths: vec![3277, 3277, 3277, 3277, 3276],
            doc_offsets: vec![0, 3277, 6554, 9831, 13_108],
            padding_locs: vec![],
        },
        PackingFixture {
            name: "skewed_packing",
            seq_len: 4096,
            doc_lengths: vec![3000, 366, 365, 365],
            doc_offsets: vec![0, 3000, 3366, 3731],
            padding_locs: vec![],
        },
        PackingFixture {
            name: "boundary_dense",
            seq_len: 4096,
            doc_lengths: vec![256; 16],
            doc_offsets: (0..16).map(|i| i * 256).collect(),
            padding_locs: vec![],
        },
        PackingFixture {
            name: "single_doc",
            seq_len: 4096,
            doc_lengths: vec![4096],
            doc_offsets: vec![0],
            padding_locs: vec![],
        },
        PackingFixture {
            name: "tail_padding",
            seq_len: 4096,
            doc_lengths: vec![1024, 1024],
            doc_offsets: vec![0, 1024],
            padding_locs: (2048..4096).collect(),
        },
    ]
}

pub fn segment_ids_from_fixture(f: &PackingFixture) -> Vec<u16> {
    let mut ids = vec![u16::MAX; f.seq_len as usize]; // padding sentinel
    for (doc_idx, (&len, &off)) in f.doc_lengths.iter().zip(f.doc_offsets.iter()).enumerate() {
        for i in 0..len as usize {
            ids[off as usize + i] = doc_idx as u16;
        }
    }
    ids
}
