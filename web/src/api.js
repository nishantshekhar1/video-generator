export async function request(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error((await res.text()) || res.statusText);
  const type = res.headers.get("content-type") || "";
  if (type.includes("application/json")) return res.json();
  return null;
}

export const STAGES = [
  "topic",
  "research",
  "script",
  "narration",
  "storyboard",
  "bible",
  "stills",
  "qa",
  "motion",
  "assemble",
];

export const STYLES = [
  ["sketch", "Toddler color-pencil"],
  ["classic_cartoon", "Classic cartoon"],
  ["storybook", "Storybook illustration"],
  ["picture_book", "Children's picture-book"],
  ["retro_digital", "Retro digital"],
  ["painterly", "Painterly"],
  ["graphic", "Graphic"],
];
