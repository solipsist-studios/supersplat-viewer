// Buffer-ahead gate for a streamed .sogst load: when the scene may appear
// and the playhead start, and how far along the progress bar is. These are
// pure functions of what the stream has measured; the stream driver owns
// the measurements and calls them.
//
// The viewer shows the scene only when the whole pipeline is ahead of
// playback. Without this test the playhead starves at segment boundaries
// through the first pass, which the viewer sees as regular hitching. The
// test has two sides, and each one carries a margin:
//
//  bytesQ: at the measured bandwidth, the remaining geometry bytes download
//          within the clip duration. This covers the network.
//  fillQ:  the standard buffering test on the measured fill rate f of
//          decoded content-time, which is buffered >= duration * (1 - f) *
//          margin. The fill rate covers network, decode and sync together,
//          so this side protects a slow CPU, where decode and not download
//          is the bottleneck.
//
// The two quotients also drive the last part of the progress bar, so the bar
// reaches 100 at the moment playback can start cleanly.

// -- buffer-ahead gate tuning --------------------------------------------
// These constants are the margins in the two readiness tests below. They are
// tuning values, not format constants. We measured them against the
// reference clips on a throttled connection. They were the smallest values
// that stopped the first playback pass from hitching at segment boundaries.
//
// Smaller margins show the scene sooner, but they also risk starving the
// playhead.

// Lowest clip duration the gate will use. Without it, a still image has
// duration 0 and the required-buffer term falls to zero.
const MIN_GATE_DURATION_S = 0.1;

// Connection setup dominates the bandwidth measured in the first fraction of
// a second. The gate therefore waits for this much time before it uses the
// estimate.
const BANDWIDTH_WARMUP_S = 0.15;

// Headroom the gate subtracts from the clip duration before it tests whether
// the remaining bytes fit. The last segment must arrive before playback
// reaches it, not at the same moment.
const DOWNLOAD_HEADROOM_S = 0.3;

// Safety factor on the bandwidth estimate. The measured throughput must
// exceed what the clip needs by this factor before the byte side of the gate
// opens.
const DOWNLOAD_MARGIN = 1.3;

// Trailing window for the fill-rate estimate. It is long enough to average
// one segment's decode, and short enough to follow a connection that gets
// slower.
const FILL_WINDOW_MS = 1500;

// Minimum wall-clock time between the oldest and the newest fill sample. The
// fill rate has no meaning below it.
const MIN_FILL_SPAN_S = 0.35;

// Lowest required buffer. Content that decodes faster than it plays still
// buffers this much before the viewer shows it.
const MIN_BUFFER_S = 0.3;

// Safety factor on the buffering inequality (buffered >= duration * (1 - f)).
const FILL_MARGIN = 1.25;

// Limit on retained fill samples. It holds FILL_WINDOW_MS of history at the
// fastest segment rate we measured, plus some margin.
const MAX_FILL_SAMPLES = 40;

type FillSample = {
    // wall-clock time, ms
    t: number;
    // buffered content time, s
    b: number;
};

// Trailing samples of wall time and buffered content, for the fill-rate
// estimate. A cumulative mean falls below the sustained rate during the
// initial decode ramp, and it then delays the reveal too long.
class FillHistory {
    readonly samples: FillSample[] = [];

    push(sample: FillSample) {
        this.samples.push(sample);
        if (this.samples.length > MAX_FILL_SAMPLES) {
            this.samples.shift();
        }
    }
}

type GateInput = {
    // archive bytes received so far
    received: number;
    // bytes up to the end of the last geometry group (meta.streams)
    geometryBytes: number;
    // seconds since the first byte
    elapsed: number;
    // clip duration, s
    duration: number;
    // content time decoded so far, relative to the clip start; Infinity
    // once all geometry is decoded
    buffered: number;
    fill: readonly FillSample[];
    // wall-clock time now, ms
    now: number;
};

type GateQuotients = { bytesQ: number; fillQ: number };

const bytesQuotient = (input: GateInput): number => {
    const { received, geometryBytes, elapsed } = input;
    if (received >= geometryBytes) {
        return 1;
    }
    if (elapsed < BANDWIDTH_WARMUP_S) {
        return 0;
    }
    const duration = Math.max(MIN_GATE_DURATION_S, input.duration);
    const bandwidth = received / elapsed;
    const target = Math.max(1, geometryBytes - (bandwidth * (duration - DOWNLOAD_HEADROOM_S)) / DOWNLOAD_MARGIN);
    return Math.min(1, received / target);
};

const fillQuotient = (input: GateInput): number => {
    const { buffered, fill, now } = input;
    if (!isFinite(buffered)) {
        return 1; // the geometry is fully decoded
    }
    if (fill.length < 2) {
        return 0;
    }
    // oldest sample inside the trailing window
    let ref = fill[0];
    for (const sample of fill) {
        if (now - sample.t <= FILL_WINDOW_MS) {
            break;
        }
        ref = sample;
    }
    const span = (now - ref.t) / 1000;
    if (buffered <= 0 || span < MIN_FILL_SPAN_S) {
        return 0;
    }
    const duration = Math.max(MIN_GATE_DURATION_S, input.duration);
    const f = Math.min(1, Math.max(0, buffered - ref.b) / span);
    const required = Math.max(MIN_BUFFER_S, duration * (1 - f) * FILL_MARGIN);
    return Math.min(1, buffered / required);
};

const gateQuotients = (input: GateInput): GateQuotients => ({
    bytesQ: bytesQuotient(input),
    fillQ: fillQuotient(input)
});

const gateOpen = ({ bytesQ, fillQ }: GateQuotients) => bytesQ >= 1 && fillQ >= 1;

// Progress before the reveal, in [0, 99]. The download up to reveal_bytes
// maps to 0-70, and buffer readiness (the lower quotient) maps to 70-99. An
// archive without geometry_bytes has no gate and maps the download to 0-99.
const streamProgress = (received: number, revealBytes: number, quotients: GateQuotients | null): number => {
    const downloaded = Math.min(1, received / revealBytes);
    if (!quotients) {
        return Math.min(99, Math.trunc(downloaded * 100));
    }
    return Math.trunc(70 * downloaded + 29 * Math.min(quotients.bytesQ, quotients.fillQ));
};

// Progress of a monolithic archive's download, which maps to 0-70; its
// decode then maps to 70-100.
const monolithicProgress = (received: number, contentLength: number): number =>
    Math.min(70, Math.trunc((received / contentLength) * 70));

export { FillHistory, gateOpen, gateQuotients, monolithicProgress, streamProgress };
export type { FillSample, GateInput, GateQuotients };
