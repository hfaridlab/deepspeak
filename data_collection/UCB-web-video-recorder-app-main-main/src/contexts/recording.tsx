import { createContext, useContext, useRef, useState } from 'react';

import { composeStreams } from 'services/composer';

import { useLayout } from './layout';
import { useStreams } from './streams';

import { generateUUID, setUniqueIdentifierCookie, getCookie, getDeviceDetails } from '../components/RecordingUtils/recordingUtils'; 

type RecordingContextType = {
  isRecording: boolean;
  isPaused: boolean;
  startRecording: () => void;
  stopRecording: () => void;
  pauseRecording: () => void;
  resumeRecording: () => void;
  randomUUID: string | null;
  setRandomUUID: (uuid: string | null) => void;
  resetUUID: () => void;
  videoBlobUrl: string | null; // Add this line
  handleRecordingComplete: (blob: Blob) => void; // Add this line
};

export const RecordingContext = createContext<RecordingContextType | undefined>(
  undefined,
);

type RecordingProviderProps = {
  children: React.ReactNode;
};

export const RecordingProvider = ({ children }: RecordingProviderProps) => {
  const { layout } = useLayout();
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const { cameraStream, microphoneStream, screenshareStream } = useStreams();

  const [randomUUID, setRandomUUID] = useState<string | null>(null);
  const mediaRecorder = useRef<MediaRecorder>();

  const resetUUID = () => {
    setRandomUUID(null);
  };

  const [videoBlobUrl, setVideoBlobUrl] = useState<string | null>(null);

  const handleRecordingComplete = (blob: Blob) => {
    const url = URL.createObjectURL(blob);
    console.log("Blob URL:", url); // Check if the URL is being logged
    setVideoBlobUrl(url);
  };
  

  const startRecording = () => {
    setIsRecording(true);

    setUniqueIdentifierCookie();

    const composedStream = composeStreams(
      layout === 'screenOnly' ? null : cameraStream,
      microphoneStream,
      layout === 'cameraOnly' ? null : screenshareStream,
    );
    mediaRecorder.current = new MediaRecorder(composedStream, {
      mimeType: 'video/webm; codecs=vp9',
      videoBitsPerSecond: 8e6,
    });

    const chunks: Blob[] = [];

    mediaRecorder.current.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };

    mediaRecorder.current.onstop = async () => {
      try {
        composedStream
          .getVideoTracks()
          .forEach((composedTrack) => composedTrack.stop());

        const blob = new Blob(chunks);
        handleRecordingComplete(blob);

        // Create a FormData object and append the blob to it
        const formData = new FormData();
        //formData.append('recording', blob, 'recording.webm');

        var randomUUID = generateUUID();
        setRandomUUID(randomUUID);
        var userId = getCookie("userId");

        const deviceDetails = await getDeviceDetails();
        
        // Capture the User-Agent string
        const userAgent = navigator.userAgent;
        //formData.append('userAgent', userAgent); // Append User-Agent to the formData
        formData.append('microphone', deviceDetails); // Append microphone details to the formData

        formData.append('recording', blob, userId + '_' + randomUUID + '.webm');
        if (userId !== null) {
          // Append userId to the formData
          formData.append('userId', userId);
        }

        try {
          // Use fetch to send the FormData to the server
          const response = await fetch('{DOMAIN}/{ENDPOINT}', { // INSERT YOUR DOMAIN AND ENDPOINTS HERE IN THE FORMAT OF <DOMAIN>/<ENDPOINT>
            method: 'POST',
            body: formData,
          });

          if (response.ok) {
            console.log('Recording successfully uploaded!');
          } else {
            console.error('Failed to upload recording.');
          }
        } catch (error) {
          console.error('Error uploading recording:', error);
        }
      } catch (error) {
        console.error('Error in onstop:', error);
      }
    };

    mediaRecorder.current.start();
  };

  const stopRecording = () => {
    mediaRecorder.current?.stop();
    setIsRecording(false);
    setIsPaused(false);
  };

  const pauseRecording = () => {
    mediaRecorder.current?.pause();
    setIsPaused(true);
  };

  const resumeRecording = () => {
    setIsPaused(false);
    mediaRecorder.current?.resume();
  };

  return (
    <RecordingContext.Provider
      value={{
        isRecording,
        isPaused,
        startRecording,
        stopRecording,
        pauseRecording,
        resumeRecording,
        randomUUID,
        setRandomUUID,
        resetUUID,
        videoBlobUrl,
        handleRecordingComplete,
      }}
    >
      {children}
    </RecordingContext.Provider>
  );
};

export const useRecording = (): RecordingContextType => {
  const context = useContext(RecordingContext);

  if (context === undefined) {
    throw new Error('useRecording must be used within a RecordingProvider');
  }

  return context;
};