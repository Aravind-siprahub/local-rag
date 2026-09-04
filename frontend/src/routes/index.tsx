import { createBrowserRouter } from 'react-router-dom'

import { AppLayout } from '@/layouts/AppLayout'
import { AuthLayout } from '@/layouts/AuthLayout'
import { Login } from '@/features/auth/pages/Login'
import { Signup } from '@/features/auth/pages/Signup'
import { ForgotPassword } from '@/features/auth/pages/ForgotPassword'
import {
  DashboardPage,
  DocumentsPage,
  NotFoundPage,
  ChatPage,
  UploadPage,
  SettingsPage,
} from '@/pages'
import { ProtectedRoute } from '@/routes/ProtectedRoute'
import { ROUTES } from '@/routes/paths'

export const appRouter = createBrowserRouter([
  // Auth pages — share the dark glassmorphism AuthLayout
  {
    element: <AuthLayout />,
    children: [
      { path: ROUTES.login,  element: <Login /> },
      { path: ROUTES.signup, element: <Signup /> },
      { path: ROUTES.forgotPassword, element: <ForgotPassword /> },
    ],
  },
  // Application Dashboard Routes
  {
    path: ROUTES.dashboard,
    element: (
      <ProtectedRoute>
        <AppLayout />
      </ProtectedRoute>
    ),
    children: [
      {
        index: true,
        element: <DashboardPage />,
      },
      {
        path: ROUTES.documents.slice(1),
        element: <DocumentsPage />,
      },
      {
        path: ROUTES.upload.slice(1),
        element: <UploadPage />,
      },
      {
        path: ROUTES.chat.slice(1),
        element: <ChatPage />,
      },
      {
        path: ROUTES.settings.slice(1),
        element: <SettingsPage />,
      },
      {
        path: '*',
        element: <NotFoundPage />,
      },
    ],
  },
])
