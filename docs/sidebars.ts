import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  apiSidebar: [
    'api/index',
    {
      type: 'category',
      label: 'Documentation',
      collapsed: false,
      items: [
        'documentation/orientation-analysis',
        'documentation/segmentation',
        'documentation/insegt',
        'documentation/volume-fraction',
        'documentation/fiber-trajectory',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      collapsed: false,
      items: [
        'api/io',
        'api/analysis',
        'api/segment',
        'api/simulation',
        'api/fiber_trajectory',
        'api/visualize',
      ],
    },
  ],
};

export default sidebars;
