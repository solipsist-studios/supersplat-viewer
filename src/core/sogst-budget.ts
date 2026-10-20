import { platform } from 'playcanvas';

import type { SogstMeta } from '../parsers/sogst';

import { shWords } from './sh-textures';

// Automatic splat budget for .sogst clips: how many splats a clip may hold
// resident before it plays through a segment window (sogst-window.ts).
//
// The device gets a memory allowance by platform tier, and each clip turns
// it into splats through an estimate of its own cost per splat, which
// depends on its SH degree and whether it has acceleration. An SH3 clip
// therefore gets fewer splats than an SH0 clip on the same device.
//
// The estimate counts what this viewer allocates for each splat. It is a
// relative cost for comparing clips, not a forecast of the page's memory:
// the browser adds its own copies and fixed costs on top. The allowances
// below are in the same units, so they are calibrated against clips known
// to play (or not) on each tier, not against an OS memory gauge.

// -- platform allowances ---------------------------------------------------
// In the cost model's bytes (see above). These are starting estimates, not
// measurements on each tier: a tier that struggles should come down, and one
// that plays everything should go up. The URL parameter sogstbudget and the
// embed option sogstBudget override them.

// iPhone, iPad and Vision Pro. Calibrated low: Vision Pro plays an 11M-splat
// SH2 clip fully resident at about 2.5 GB by this model, and the phones
// share the platform with far less memory.
const APPLE_TOUCH_ALLOWANCE = 1.5e9;

// Android and other mobile devices.
const MOBILE_ALLOWANCE = 1e9;

// Desktops and laptops. Sized for an 8 GB laptop with integrated graphics,
// the smallest common desktop: an 11M-splat SH3 clip fully resident (about
// 3.2 GB by this model) crashed the tab on one, while it plays windowed.
const DESKTOP_ALLOWANCE = 1.5e9;

// -- per-splat cost model --------------------------------------------------

// Stream textures every splat has: splatColor (RGBA16F, 8 bytes),
// transformA (RGBA32U, 16) and transformB (RGBA16F, 8).
const BASE_STREAM_BYTES = 32;

// splatMotion (RGBA32F, 16) and splatTemporal (R32F, 4).
const MOTION_STREAM_BYTES = 20;

// splatAccel (RGBA32F), in clips that have acceleration.
const ACCEL_STREAM_BYTES = 16;

// Each stream exists twice: as the GPU texture, and as the CPU copy that
// row updates write into before uploading (gsplat-range-sync.ts).
const STREAM_COPIES = 2;

// The renderer's own per-splat state: the resource's centers (3 float32,
// 12 bytes), the unified renderer's work buffer (~36), and the sort order.
const RENDERER_BYTES = 56;

// Bytes per splat a clip costs, by the model above.
const sogstSplatBytes = (meta: SogstMeta): number => {
    const sh = 4 * shWords(meta.shN?.bands ?? 0);
    const streams = BASE_STREAM_BYTES + sh + MOTION_STREAM_BYTES + (meta.accel ? ACCEL_STREAM_BYTES : 0);
    return streams * STREAM_COPIES + RENDERER_BYTES;
};

type SogstBudgetTier = 'apple-touch' | 'mobile' | 'desktop';

// Apple's touch devices report themselves in different ways: iPhone as
// iOS, while iPadOS and visionOS Safari send a desktop macOS user agent,
// which the engine's visionos test (a Mac with touch points) catches, iPads
// included.
const sogstBudgetTier = (): SogstBudgetTier => {
    if (platform.ios || platform.visionos) {
        return 'apple-touch';
    }
    return platform.mobile ? 'mobile' : 'desktop';
};

const ALLOWANCES: Record<SogstBudgetTier, number> = {
    'apple-touch': APPLE_TOUCH_ALLOWANCE,
    mobile: MOBILE_ALLOWANCE,
    desktop: DESKTOP_ALLOWANCE
};

const millions = (splats: number) => `${(splats / 1e6).toFixed(2)}M`;

// The splat budget for a clip: the host's override when it gave one (0 for
// no budget), otherwise the automatic one. Logs the choice and its source,
// so a device's console shows which budget a load used.
const sogstBudgetFor = (meta: SogstMeta, override: number | undefined): number => {
    const clip = `clip has ${millions(meta.count)}`;
    if (override !== undefined) {
        console.debug(
            override > 0
                ? `sogst: budget ${millions(override)} splats (sogstbudget override; ${clip})`
                : `sogst: no budget, fully resident (sogstbudget=0; ${clip})`
        );
        return override;
    }
    const tier = sogstBudgetTier();
    const bytes = sogstSplatBytes(meta);
    const budget = Math.floor(ALLOWANCES[tier] / bytes);
    console.debug(`sogst: automatic budget ${millions(budget)} splats (${tier} tier, ~${bytes} B/splat; ${clip})`);
    return budget;
};

export { sogstBudgetFor, sogstBudgetTier, sogstSplatBytes };
