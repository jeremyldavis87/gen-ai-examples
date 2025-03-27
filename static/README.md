# Static Assets

This directory contains static assets for the gen-ai-examples project, following AWS best practices for front-end development with React and AWS Amplify Gen 2.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Asset Types](#asset-types)
4. [Usage](#usage)
5. [AWS S3 Integration](#aws-s3-integration)
6. [AWS Amplify Integration](#aws-amplify-integration)

## Overview

The static directory contains assets used by the front-end applications in the gen-ai-examples project. These assets include images, CSS files, JavaScript libraries, and other resources that are served to the client without modification.

## Directory Structure

The static assets are organized into the following structure:

```
static/
├── css/         # Stylesheets
├── images/       # Image files
├── js/           # JavaScript files
├── fonts/        # Font files
└── templates/    # HTML templates
```

## Asset Types

### Images

The `images/` directory contains image assets used by the applications, including:

- Logos and brand assets
- UI icons and graphics
- Background images
- Placeholder images for testing

### CSS

The `css/` directory contains stylesheet assets, including:

- Base styles and resets
- Component-specific styles
- Responsive design rules
- Animation definitions

### JavaScript

The `js/` directory contains JavaScript assets, including:

- Utility functions
- Third-party libraries
- Custom components
- Polyfills and shims

### Fonts

The `fonts/` directory contains font assets used for consistent typography across the applications.

### Templates

The `templates/` directory contains HTML templates used by the applications.

## Usage

### In FastAPI Applications

In FastAPI applications, static assets are served using the `StaticFiles` middleware:

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Mount the static directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
```

Then, you can reference static assets in your templates or API responses:

```html
<img src="/static/images/logo.png" alt="Logo">
<link rel="stylesheet" href="/static/css/styles.css">
<script src="/static/js/app.js"></script>
```

### In React Applications

In React applications, static assets are imported directly in your components:

```jsx
import React from 'react';
import logo from '../static/images/logo.png';
import '../static/css/styles.css';

function App() {
  return (
    <div className="app">
      <img src={logo} alt="Logo" />
      <h1>Welcome to the App</h1>
    </div>
  );
}

export default App;
```

## AWS S3 Integration

Static assets can be deployed to Amazon S3 for efficient delivery and scaling:

```bash
# Sync static assets to S3
aws s3 sync static/ s3://your-bucket-name/static/ --acl public-read
```

You can then reference the assets using the S3 URL:

```
https://your-bucket-name.s3.amazonaws.com/static/images/logo.png
```

For production environments, it's recommended to use Amazon CloudFront with S3 to provide caching and edge delivery of static assets.

## AWS Amplify Integration

When deploying with AWS Amplify Gen 2, static assets are automatically handled as part of the build process:

### Amplify Configuration

In your `amplify.yml` file, ensure that static assets are included in the build output:

```yaml
version: 1
frontend:
  phases:
    preBuild:
      commands:
        - npm install
    build:
      commands:
        - npm run build
  artifacts:
    baseDirectory: build
    files:
      - '**/*'
  cache:
    paths:
      - node_modules/**/*
```

### Accessing Assets in Amplify

In your React components, you can access static assets using relative paths:

```jsx
import React from 'react';

function Logo() {
  return <img src="/static/images/logo.png" alt="Logo" />;
}

export default Logo;
```

Amplify will automatically handle the serving of these assets from the appropriate location.

For more information on managing static assets with AWS Amplify, refer to the [AWS Amplify Documentation](https://docs.amplify.aws/guides/hosting/static-assets/q/platform/js/).
