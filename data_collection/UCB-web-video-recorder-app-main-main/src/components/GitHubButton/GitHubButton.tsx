import GitHubIcon from '@mui/icons-material/GitHub';
import styles from './GitHubButton.module.css';

const GitHubButton = () => {
  return (
    <div className={styles.root}>
      <span className={styles.iconPlaceholder}></span>
      <span className={styles.textPlaceholder}></span>
    </div>
  );
};

export default GitHubButton;