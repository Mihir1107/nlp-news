import React from 'react'
import ReactDOM from 'react-dom/client'
import { MotionConfig } from 'framer-motion'
// Fonts are bundled into the build, so the app works without internet during a demo.
import '@fontsource-variable/bodoni-moda/opsz.css'         // display: nameplate, headlines, figures
import '@fontsource-variable/bodoni-moda/opsz-italic.css'
import '@fontsource/newsreader/400.css'                      // text and interface
import '@fontsource/newsreader/400-italic.css'
import '@fontsource/newsreader/500.css'
import '@fontsource/newsreader/600.css'
import '@fontsource/newsreader/600-italic.css'
import './styles.css'
import App from './App.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {/* reducedMotion="user": people who ask their OS for less motion get instant transitions */}
    <MotionConfig reducedMotion="user">
      <App />
    </MotionConfig>
  </React.StrictMode>,
)
