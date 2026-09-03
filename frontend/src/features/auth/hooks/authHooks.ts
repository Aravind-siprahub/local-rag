import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { authApi } from '../api/authApi'
import type { LoginCredentials, SignupCredentials, User } from '../types/authTypes'
import { AuthStore } from '../utils/authStore'
import { ROUTES } from '@/routes/paths'

export const AUTH_QUERY_KEY = ['auth', 'user']

/**
 * Hook for managing current Auth state & session
 */
export function useAuth() {
  const [user, setUser] = useState<User | null>(() => AuthStore.getUser())
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => AuthStore.isAuthenticated())

  useEffect(() => {
    const handleAuthChange = () => {
      setUser(AuthStore.getUser())
      setIsAuthenticated(AuthStore.isAuthenticated())
    }

    window.addEventListener('auth:unauthorized', handleAuthChange)
    window.addEventListener('auth:change', handleAuthChange)
    return () => {
      window.removeEventListener('auth:unauthorized', handleAuthChange)
      window.removeEventListener('auth:change', handleAuthChange)
    }
  }, [])

  const logout = () => {
    AuthStore.clearSession()
    setUser(null)
    setIsAuthenticated(false)
    window.dispatchEvent(new Event('auth:change'))
  }

  return {
    user,
    isAuthenticated,
    accessToken: AuthStore.getAccessToken(),
    logout,
  }
}

/**
 * Custom Hook for User Login Mutation
 */
export function useLogin() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()

  return useMutation({
    mutationFn: async (credentials: LoginCredentials) => {
      const data = await authApi.login(credentials)
      if (data.auth) {
        AuthStore.setSession(data.auth.accessToken, data.auth.user)
        AuthStore.setRememberedEmail(credentials.email, Boolean(credentials.rememberMe))
      }
      return data
    },
    onSuccess: (data) => {
      if (data.auth) {
        queryClient.setQueryData(AUTH_QUERY_KEY, data.auth.user)
        navigate(ROUTES.dashboard, { replace: true })
      }
    },
  })
}

/**
 * Custom Hook for User Sign Up / Registration Mutation
 */
export function useRegister() {
  return useMutation({
    mutationFn: async (credentials: SignupCredentials) => {
      return await authApi.register(credentials)
    },
  })
}


/**
 * Custom Hook for User Logout
 */
export function useLogout() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()

  return useMutation({
    mutationFn: async () => {
      await authApi.logout()
      AuthStore.clearSession()
    },
    onSettled: () => {
      queryClient.removeQueries({ queryKey: AUTH_QUERY_KEY })
      navigate('/login', { replace: true })
    },
  })
}

/**
 * Hook to Fetch Current User Profile from API
 */
export function useCurrentUser() {
  return useQuery({
    queryKey: AUTH_QUERY_KEY,
    queryFn: async () => {
      if (!AuthStore.getAccessToken()) return null
      return authApi.getCurrentUser()
    },
    enabled: AuthStore.isAuthenticated(),
    initialData: AuthStore.getUser() || undefined,
  })
}
