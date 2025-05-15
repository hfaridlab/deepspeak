import cx from 'classnames';
import React, { useContext } from 'react';

import Footer from 'components/Footer';
import CameraSelect from 'components/CameraSelect';
import MicrophoneSelect from 'components/MicrophoneSelect';
import LayoutSwitcher from 'components/LayoutSwitcher';
import PiPWindow from 'components/PiPWindow';
import VideoStreams from 'components/VideoStreams';
import { useLayout } from 'contexts/layout';
import { useMediaDevices } from 'contexts/mediaDevices';
import { usePictureInPicture } from 'contexts/pictureInPicture';
import { useStreams } from 'contexts/streams';
import useKeyboardShorcut from 'hooks/useKeyboardShortcut';

import MainRecordButton from 'components/MainRecordButton';
import { RandomUUIDProvider } from 'contexts/RandomUUIDContext';
import { RecordingContext } from 'contexts/recording';

import styles from './App.module.css';

const App = () => {
  const { layout } = useLayout();
  const { cameraStream, screenshareStream } = useStreams();
  const { pipWindow } = usePictureInPicture();
  const { randomUUID } = useContext(RecordingContext); // Access the randomUUID from context

  const { videoBlobUrl } = useContext(RecordingContext);
  
  const {
    cameraEnabled,
    microphoneEnabled,
    setCameraEnabled,
    setMicrophoneEnabled,
  } = useMediaDevices();

  useKeyboardShorcut('e', () => setCameraEnabled(!cameraEnabled));
  useKeyboardShorcut('d', () => setMicrophoneEnabled(!microphoneEnabled));

  const copyToClipboard = () => {
    navigator.clipboard.writeText(randomUUID)
      .then(() => {
        console.log('UUID copied to clipboard!');
        // You can also implement a notification to the user here
      })
      .catch(err => {
        console.error('Failed to copy UUID: ', err);
      });
  };

  const downloadVideo = () => {
    // Create a link and set the URL and download name
    const link = document.createElement('a');
    link.href = videoBlobUrl;
    link.download = `Recording-${randomUUID || 'video'}.webm`; // You can customize the filename format

    // Append to the DOM, click to download, and then remove it
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };


  return (
    <RandomUUIDProvider>
    <div
      className={cx(styles.root, {
        [styles.placeholder]:
          layout === 'cameraOnly' ? !cameraStream : !screenshareStream,
      })}
    >
      {/* Text above the video */}
      <div className={styles.textAboveVideo}>
        <p><strong>Video Recorder App</strong></p>
        <p>Please use this webpage to record yourself responding to the prompts given to you in the study.</p>
        <p>A preview of your camera can be seen below.</p>
        <ul className={styles.roundBullets}>
          <li>Launch the recording window by clicking 'launch recorder' in the preview below.</li>
          <li>In the recording window, click the record button when you are ready to record.</li>
          <li>To stop the recording, press the stop recording button.</li>
          </ul>
        <p>Once you have completed recording:</p>
        <ul className={styles.roundBullets}>
          <li>A 'recording identifier' will be provided below this text in <span style={{ color: 'blue' }}>blue</span>. Copy this back into your survey response.</li>
          <li>You can re-record yourself as many times as you wish, but only paste the <b>final</b> identifier into the survey response.</li>
          <li>There will be an option to download and playback the video to check it worked properly.</li>
        </ul>
      </div>
      {/* End of text */}
      {randomUUID && (
        <p>
        Recording Identifier: <span style={{ color: 'blue' }}>{randomUUID}</span>
        {' '} {/* This adds a space */}
        <button onClick={copyToClipboard}>Copy to Clipboard</button>
        {videoBlobUrl && (
              <button onClick={downloadVideo} style={{ marginLeft: '10px' }}>Download Last Video</button>
            )}
      </p>
      )}

      <main className={styles.main}>
        {/* VideoStreams component with adjusted size */}
        <div className={styles.videoContainer}>
          <VideoStreams />
        </div>

        {/* Record button or other components */}
        <div className={styles.recordButton}>
          <MainRecordButton />
        </div>

        {/* Camera and Microphone Selectors */}
        <div className={styles.deviceSelectors}>
            <MicrophoneSelect />
            <CameraSelect />
          </div>

        <LayoutSwitcher />
      </main>
    

      {/* Footer component */}
      {/*<Footer />*/}

      {/* PiPWindow component */}
      {pipWindow && <PiPWindow pipWindow={pipWindow} />}
    </div>
    </RandomUUIDProvider>
  );
};


export default App;
