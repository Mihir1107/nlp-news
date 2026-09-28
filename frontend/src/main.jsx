import React from 'react'
import ReactDOM from 'react-dom/client'
import { MotionConfig } from 'framer-motion'
// Fonts are bundled into the build, so the app works without internet during a demo.
import '@fontsource-variable/bodoni-moda/opsz.css'         // display: nameplate, headlines, figures
import '@fontsource-variable/bodoni-moda/opsz-italic.css'
import '@fontsource-variable/newsreader/opsz.css'          // text and interface; its optical-size axis
import '@fontsource-variable/newsreader/opsz-italic.css'   // gives captions sturdier letters than headlines
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
