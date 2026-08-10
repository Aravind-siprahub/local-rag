import React from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '@/features/auth/hooks/authHooks'
import { ROUTES } from '@/routes/paths'

interface ProtectedRouteProps {
  children: React.ReactNode
  requireAuth?: boolean
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requireAuth = true,
}) => {
  const { isAuthenticated } = useAuth()
  const location = useLocation()

  if (requireAuth && !isAuthenticated) {
    // Redirect unauthenticated user to login page
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  if (!requireAuth && isAuthenticated) {
    // Redirect authenticated user away from auth pages to dashboard
    return <Navigate to={ROUTES.dashboard} replace />
  }

  return <>{children}</>
}
