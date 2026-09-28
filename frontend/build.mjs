// Bundle the React app into dist/ with esbuild: JSX -> JavaScript, CSS and font files copied alongside.
// FastAPI serves dist/ at http://127.0.0.1:8000/ (see api/app.py).
//   npm run build   one production build (minified)
//   npm run watch   rebuild on every file change while developing; reload the browser to see changes
import * as esbuild from 'esbuild'
import { cp, mkdir, rm } from 'node:fs/promises'

const watch = process.argv.includes('--watch')
await rm('dist', { recursive: true, force: true })
await mkdir('dist', { recursive: true })
await cp('index.html', 'dist/index.html')

const ctx = await esbuild.context({
  entryPoints: ['src/main.jsx'],
  bundle: true,                       // follow imports and put everything into one file
  outdir: 'dist/assets',
  entryNames: 'app',                  // -> dist/assets/app.js and dist/assets/app.css
  assetNames: 'fonts/[name]-[hash]',
  loader: { '.woff2': 'file', '.woff': 'file' },
  jsx: 'automatic',                   // React 17+ JSX transform, no `import React` needed
  minify: !watch,
  sourcemap: watch,
  target: ['es2020'],
  define: { 'process.env.NODE_ENV': JSON.stringify(watch ? 'development' : 'production') },
  logLevel: 'info',
})
if (watch) {
  await ctx.watch()
} else {
  await ctx.rebuild()
  await ctx.dispose()
}
