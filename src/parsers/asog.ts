// .asog: Animated SOG, the compact 4D clip format of animated-sog (github.com/arrival-space/animated-sog, MIT).
// The file is decoded and played by that package's `AsogPlayer` (core/load-asog.ts); this only names the format.

const isAsogFilename = (filename: string): boolean => filename.toLowerCase().endsWith('.asog');

export { isAsogFilename };
