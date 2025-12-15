import React from 'react';
import ReactDOM from 'react-dom/client';

console.log('index.tsx: Starting execution');

// Import the test component instead of the main App
import TestComponent from './src/test';

console.log('index.tsx: Test component imported successfully');

const rootElement = document.getElementById('root');
console.log('index.tsx: Root element found:', rootElement);

if (!rootElement) {
  console.error("index.tsx: Could not find root element to mount to");
  throw new Error("Could not find root element to mount to");
}

console.log('index.tsx: Creating React root');
const root = ReactDOM.createRoot(rootElement);

console.log('index.tsx: Rendering Test component');
root.render(
  <TestComponent />
);

console.log('index.tsx: Test component rendered successfully');