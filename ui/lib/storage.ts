const API_KEY_STORAGE_KEY = "rag-ui:api-key";
const API_URL_STORAGE_KEY = "rag-ui:api-url";
const SELECTED_COLLECTION_STORAGE_KEY = "rag-ui:selected-collection";
const UPLOADED_DOCS_STORAGE_KEY = "rag-ui:uploaded-docs";

export const DEFAULT_API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function getStoredApiKey(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(API_KEY_STORAGE_KEY) ?? "";
}

export function setStoredApiKey(value: string): void {
  window.localStorage.setItem(API_KEY_STORAGE_KEY, value);
}

export function getStoredApiUrl(): string {
  if (typeof window === "undefined") return DEFAULT_API_URL;
  return window.localStorage.getItem(API_URL_STORAGE_KEY) ?? DEFAULT_API_URL;
}

export function setStoredApiUrl(value: string): void {
  window.localStorage.setItem(API_URL_STORAGE_KEY, value);
}

export function getStoredSelectedCollectionId(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(SELECTED_COLLECTION_STORAGE_KEY);
}

export function setStoredSelectedCollectionId(id: string | null): void {
  if (id === null) {
    window.localStorage.removeItem(SELECTED_COLLECTION_STORAGE_KEY);
  } else {
    window.localStorage.setItem(SELECTED_COLLECTION_STORAGE_KEY, id);
  }
}

export type TrackedDocument = {
  document_id: string;
  filename: string;
  collection_id: string;
};

export function getTrackedDocuments(): TrackedDocument[] {
  if (typeof window === "undefined") return [];
  const raw = window.localStorage.getItem(UPLOADED_DOCS_STORAGE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function addTrackedDocument(doc: TrackedDocument): void {
  const docs = getTrackedDocuments();
  docs.unshift(doc);
  window.localStorage.setItem(UPLOADED_DOCS_STORAGE_KEY, JSON.stringify(docs));
}
