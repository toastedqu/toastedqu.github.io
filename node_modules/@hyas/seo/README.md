# Hyas SEO

Official SEO integration for Hyas.

## Status

[![npm (scoped)](https://img.shields.io/npm/v/@hyas/seo?style=flat-square)](https://www.npmjs.com/package/@hyas/seo)

## Installation

```bash
npm install @hyas/seo@latest
```

## Setup

Add mounts to `./config/_default/module.toml`:

```toml
[[mounts]]
  source = "node_modules/@hyas/seo/layouts"
  target = "layouts"

[[mounts]]
  source = "layouts"
  target = "layouts"
```

Add settings to `./config/_default/hugo.toml`:

```toml
title = "Hyas"
enableRobotsTXT = true
```

Set parameters in `./config/_default/params.toml`:

```toml
# Hugo
title = "Hyas"
description = "Congrats on setting up a new Doks project!"
images = ["cover.png"]

[social]
  twitter = "getdoks"
  facebook_admin = ""

# SEO (@hyas/seo)
[seo]
  [seo.title]
    separator = "" # "|" (default)
    suffix = "" # title hugo.toml (default)
  [seo.favicons]
    icon = "favicon.png" # favicon.png (default)
    svgIcon = "favicon.svg" # favicon.svg (default)
  [seo.schemas]
    type = "Organization" # Organization (default) or Person
    logo = "favicon-512x512.png" # Logo of Organization — favicon-512x512.png (default)
    name = "Hyas" # Name of Organization or Person
    sameAs = [] # E.g. ["https://github.com/gethyas/hyas", "https://fosstodon.org/@hyas"]
    images = ["cover.png"] # ["cover.png"] (default)
    article = [] # Article sections
    newsArticle = [] # NewsArticle sections
    blogPosting = ["blog"] # BlogPosting sections
    product = [] # Product sections
```

Set parameters in page frontmatter:

```yml
seo:
  title: "" # custom title (optional)
  description: "" # custom description (recommended)
  canonical: "" # custom canonical URL (optional)
  robots: "" # custom robot tags (optional)
  structured_data:
    product:
      currency: USD
      price: 500
      availability: https://schema.org/OnlineOnly
```

## How to use

See the Hyas SEO documentation:

- [Hyas SEO](https://seo.gethyas.com/)

## Credits

This npm package is based on:

- [Hugo SEO](https://gitlab.com/hugo-modules/hugo-seo)
- [Structured Data for Breadcrumbs](https://bullaki.com/projects/web-design/seo-with-hugo-5-breadcrumbs/)
- [Schema.org pieces](https://developer.yoast.com/features/schema/pieces/)
