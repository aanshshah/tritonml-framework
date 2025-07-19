# TritonML Documentation Site

This folder contains the GitHub Pages documentation site for TritonML.

## Local Development

To run the site locally:

1. Install Jekyll (if not already installed):
   ```bash
   gem install bundler jekyll
   ```

2. Navigate to the docs folder:
   ```bash
   cd docs
   ```

3. Install dependencies:
   ```bash
   bundle install
   ```

4. Run the development server:
   ```bash
   bundle exec jekyll serve
   ```

5. Open http://localhost:4000 in your browser

## Structure

```
docs/
├── index.html          # Main landing page
├── api.html           # API documentation
├── _config.yml        # Jekyll configuration
├── assets/
│   ├── css/
│   │   ├── style.css  # Main styles
│   │   └── api.css    # API docs styles
│   ├── js/
│   │   └── main.js    # Interactive features
│   └── img/
│       └── favicon.svg # Site favicon
└── README.md          # This file
```

## Deployment

The site is automatically deployed to GitHub Pages when you push to the `main` branch. Make sure GitHub Pages is enabled in your repository settings and set to deploy from the `/docs` folder.

## Customization

1. **Update URLs**: Replace `yourusername` with your GitHub username in:
   - `index.html`
   - `api.html`
   - `_config.yml`

2. **Analytics**: Add your Google Analytics ID in `_config.yml`

3. **Social Links**: Update social media links in the footer

4. **Colors**: Modify CSS variables in `style.css` to change the color scheme

## Features

- Responsive design
- Syntax highlighting with Prism.js
- Smooth scrolling navigation
- Interactive code examples
- Copy-to-clipboard functionality
- Mobile-friendly navigation
- SEO optimized