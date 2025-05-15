// RandomUUIDContext.tsx
import { createContext, useContext, useState, ReactNode } from 'react';

type RandomUUIDContextType = {
  randomUUID: string | null;
  setRandomUUID: React.Dispatch<React.SetStateAction<string | null>>;
};

const RandomUUIDContext = createContext<RandomUUIDContextType | undefined>(
  undefined
);

type RandomUUIDProviderProps = {
  children: ReactNode;
};

export const RandomUUIDProvider = ({ children }: RandomUUIDProviderProps) => {
  const [randomUUID, setRandomUUID] = useState<string | null>(null);

  return (
    <RandomUUIDContext.Provider value={{ randomUUID, setRandomUUID }}>
      {children}
    </RandomUUIDContext.Provider>
  );
};

export const useRandomUUID = (): RandomUUIDContextType => {
  const context = useContext(RandomUUIDContext);

  if (!context) {
    throw new Error('useRandomUUID must be used within a RandomUUIDProvider');
  }

  return context;
};
