import style from './App.module.less';
import { BrowserRouter } from 'react-router-dom';
import RouterRoutes from '@/routes/routes';
import vconsole from 'vconsole';
import { useEffect } from 'react';
import ChatProvider from '@/context/chat-context';

function App() {
    useEffect(() => {
        new vconsole();
    }, []);

    return (
        <BrowserRouter>
            <ChatProvider>
                <div className={style.app} id="app">
                    <RouterRoutes />
                </div>
            </ChatProvider>
        </BrowserRouter>
    );
}

export default App;
