const videos = [document.querySelector('#gt-video'), document.querySelector('#ours-video')];
const toggle = document.querySelector('#play-toggle');
const timeline = document.querySelector('#timeline');
const time = document.querySelector('#time');
let seeking = false;

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? seconds : 0;
  return `${String(Math.floor(safe / 60)).padStart(2, '0')}:${String(Math.floor(safe % 60)).padStart(2, '0')}`;
}

toggle.addEventListener('click', async () => {
  const shouldPlay = videos[0].paused;
  if (shouldPlay) await Promise.all(videos.map(video => video.play()));
  else videos.forEach(video => video.pause());
  toggle.textContent = shouldPlay ? 'Pause' : 'Play';
  toggle.setAttribute('aria-label', `${shouldPlay ? 'Pause' : 'Play'} comparison videos`);
});

timeline.addEventListener('input', () => {
  seeking = true;
  const duration = videos[0].duration || 0;
  videos.forEach(video => { video.currentTime = duration * Number(timeline.value) / 1000; });
});
timeline.addEventListener('change', () => { seeking = false; });
videos[0].addEventListener('timeupdate', () => {
  if (!seeking && videos[0].duration) timeline.value = Math.round(videos[0].currentTime / videos[0].duration * 1000);
  time.textContent = formatTime(videos[0].currentTime);
  if (Math.abs(videos[1].currentTime - videos[0].currentTime) > 0.12) videos[1].currentTime = videos[0].currentTime;
});

const comparisonSlider = document.querySelector('#comparison-slider');
const ellipsoidComparison = document.querySelector('#ellipsoid-compare');
comparisonSlider.addEventListener('input', () => {
  ellipsoidComparison.style.setProperty('--position', `${comparisonSlider.value}%`);
});
