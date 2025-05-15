import CameraSelect from 'components/CameraSelect';
import GitHubButton from 'components/GitHubButton';

import MicrophoneSelect from 'components/MicrophoneSelect';

import styles from './Footer.module.css';

const Footer = () => {
  return (
    <footer className={styles.root}>
      <GitHubButton />

      <div className={styles.devices}>
        <MicrophoneSelect />
        <CameraSelect />
      </div>
    </footer>
  );
};

export default Footer;
