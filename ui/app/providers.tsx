"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import {
  DEFAULT_API_URL,
  getStoredApiKey,
  getStoredApiUrl,
  getStoredSelectedCollectionId,
  setStoredApiKey,
  setStoredApiUrl,
  setStoredSelectedCollectionId,
} from "@/lib/storage";

type AppConfig = {
  apiKey: string;
  apiUrl: string;
  selectedCollectionId: string | null;
  selectedCollectionName: string | null;
  hydrated: boolean;
  setApiKey: (value: string) => void;
  setApiUrl: (value: string) => void;
  setSelectedCollection: (id: string | null, name: string | null) => void;
};

const AppConfigContext = createContext<AppConfig | null>(null);

export function AppConfigProvider({ children }: { children: ReactNode }) {
  const [apiKey, setApiKeyState] = useState("");
  const [apiUrl, setApiUrlState] = useState(DEFAULT_API_URL);
  const [selectedCollectionId, setSelectedCollectionId] = useState<string | null>(null);
  const [selectedCollectionName, setSelectedCollectionName] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setApiKeyState(getStoredApiKey());
    setApiUrlState(getStoredApiUrl());
    setSelectedCollectionId(getStoredSelectedCollectionId());
    setSelectedCollectionName(window.localStorage.getItem("rag-ui:selected-collection-name"));
    setHydrated(true);
  }, []);

  const setApiKey = (value: string) => {
    setApiKeyState(value);
    setStoredApiKey(value);
  };

  const setApiUrl = (value: string) => {
    setApiUrlState(value);
    setStoredApiUrl(value);
  };

  const setSelectedCollection = (id: string | null, name: string | null) => {
    setSelectedCollectionId(id);
    setSelectedCollectionName(name);
    setStoredSelectedCollectionId(id);
    if (name === null) {
      window.localStorage.removeItem("rag-ui:selected-collection-name");
    } else {
      window.localStorage.setItem("rag-ui:selected-collection-name", name);
    }
  };

  return (
    <AppConfigContext.Provider
      value={{
        apiKey,
        apiUrl,
        selectedCollectionId,
        selectedCollectionName,
        hydrated,
        setApiKey,
        setApiUrl,
        setSelectedCollection,
      }}
    >
      {children}
    </AppConfigContext.Provider>
  );
}

export function useAppConfig(): AppConfig {
  const ctx = useContext(AppConfigContext);
  if (!ctx) throw new Error("useAppConfig must be used within AppConfigProvider");
  return ctx;
}
