"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { fetchCurrentUser, loginUser, registerUser, type AuthUser } from "@/lib/api";
import { readStorage, removeStorage, writeStorage } from "@/lib/storage";

export type User = AuthUser;

type AuthContextValue = {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isReady: boolean;
  login: (credentials: { email: string; password: string }) => Promise<void>;
  register: (credentials: { name: string; email: string; password: string }) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(() =>
    readStorage<User | null>("next-ecomm-user", null),
  );
  const [token, setToken] = useState<string | null>(() =>
    readStorage<string | null>("next-ecomm-token", null),
  );
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    let active = true;
    if (!token) {
      setIsReady(true);
      return;
    }

    fetchCurrentUser(token)
      .then((nextUser) => {
        if (!active) return;
        setUser(nextUser);
        writeStorage("next-ecomm-user", nextUser);
      })
      .catch(() => {
        if (!active) return;
        logout();
      })
      .finally(() => {
        if (active) setIsReady(true);
      });

    return () => {
      active = false;
    };
  }, []);

  function storeSession(nextUser: User, nextToken: string) {
    setUser(nextUser);
    setToken(nextToken);
    writeStorage("next-ecomm-user", nextUser);
    writeStorage("next-ecomm-token", nextToken);
  }

  async function login(credentials: { email: string; password: string }) {
    const response = await loginUser(credentials);
    storeSession(response.user, response.access_token);
  }

  async function register(credentials: { name: string; email: string; password: string }) {
    const response = await registerUser(credentials);
    storeSession(response.user, response.access_token);
  }

  function logout() {
    setUser(null);
    setToken(null);
    removeStorage("next-ecomm-user");
    removeStorage("next-ecomm-token");
  }

  const value = useMemo(
    () => ({
      user,
      token,
      isAuthenticated: Boolean(token),
      isReady,
      login,
      register,
      logout,
    }),
    [isReady, token, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
