use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub enum EventKind {
    Arrival(usize),
    ServiceCompletion,
    SwitchoverCompletion(usize),
}

#[derive(Debug, Clone, Copy)]
pub struct Event {
    pub time: f64,
    pub seq: u64,
    pub kind: EventKind,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.seq == other.seq
    }
}
impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; invert so the earliest time comes out first.
        other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}
