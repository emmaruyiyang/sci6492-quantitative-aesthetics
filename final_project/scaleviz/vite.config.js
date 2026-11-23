export default {
  // Relative asset URLs, so the build works wherever it is mounted: a domain
  // root, a GitHub Pages project path (username.github.io/repo/), or a file
  // served out of a subfolder. The default './' -> absolute '/assets/...'
  // 404s under any subpath. Every runtime fetch already goes through
  // import.meta.env.BASE_URL (see src/main.js), so the data files follow.
  base: './',
  build: { target: 'esnext' },
  esbuild: { target: 'esnext' },
};
