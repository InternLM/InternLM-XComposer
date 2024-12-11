import Home from '@/pages/home/home';
import { ReactNode } from 'react';
import { Navigate, useRoutes } from 'react-router-dom';

interface RouteItem {
    path: string;
    element: ReactNode;
}

const routes: RouteItem[] = [
    {
        path: '/',
        element: <Home />,
    },
    {
        path: '*',
        element: <Navigate to="/" />,
    },
];

const WrapperRoutes = () => {
    return useRoutes(routes);
};

export default WrapperRoutes;
