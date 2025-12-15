import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'VMM-FRC',
  tagline: 'Virtual Microstructure Modeling for Fiber Reinforced Polymer Composites',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://naruki-ichihara.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  baseUrl: process.env.NODE_ENV === 'development' ? '/' : '/vmm_frc/',

  // GitHub pages deployment config.
  organizationName: 'Naruki-Ichihara',
  projectName: 'vmm_frc',

  onBrokenLinks: 'throw',

  markdown: {
    parseFrontMatter: async (params) => {
      const result = await params.defaultParseFrontMatter(params);
      return result;
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/Naruki-Ichihara/vmm-frc/tree/main/docs/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          routeBasePath: '/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig: {
    image: 'img/vmm-frc-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
    navbar: {
      title: 'VMM-FRC',
      logo: {
        alt: 'VMM-FRC Logo',
        src: 'img/logo.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'apiSidebar',
          position: 'left',
          label: 'API Reference',
        },
        {
          href: 'https://github.com/Naruki-Ichihara/vmm-frc',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'API',
          items: [
            {
              label: 'API Reference',
              to: '/',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'PyVista',
              href: 'https://pyvista.org/',
            },
            {
              label: 'scikit-image',
              href: 'https://scikit-image.org/',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/Naruki-Ichihara/vmm-frc',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} VMM-FRC. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'yaml'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
