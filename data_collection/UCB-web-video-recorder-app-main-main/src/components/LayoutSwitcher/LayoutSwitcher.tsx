import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';

import { Layout, useLayout } from 'contexts/layout';

import CameraOnlyIcon from './icons/CameraOnlyIcon';

import styles from './LayoutSwitcher.module.css';

const LayoutSwitcher = () => {
  const { layout, setLayout } = useLayout();

  // Determine whether to render the Camera Only button or not (e.g., based on a certain condition)
  const shouldRenderCameraOnlyButton = false; // Replace with your condition

  return (
    <ToggleButtonGroup
      className={styles.root}
      exclusive
      value={layout}
      onChange={(_, layout: Layout | null) => {
        if (layout !== null) {
          setLayout(layout);
        }
      }}
    >
      {shouldRenderCameraOnlyButton && (
        <ToggleButton value="cameraOnly">
          <CameraOnlyIcon />
          Camera only
        </ToggleButton>
      )}
    </ToggleButtonGroup>
  );
};

export default LayoutSwitcher;
