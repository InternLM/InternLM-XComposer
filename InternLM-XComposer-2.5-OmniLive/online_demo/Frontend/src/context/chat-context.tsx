import { createContext, useEffect, useMemo, useState } from 'react';

interface ChatContext {
    sessionId: string;
    status: IChatStatus;
    setStatus: (state: IChatStatus) => void;
}

export enum IChatStatus {
    DEFAULT = 1, // default
    PREPARING = 2, // after websocket onopen, before receiving the first message
    LISTENING = 3, // after receiving the first message
    OUTPUTING = 4, // after receiving @@voice_start
    FORCE_CLOSE = 5, // socket closed passively
    SOCKET_ERROR = 6, // socket error
}
export const ChatContext = createContext({} as ChatContext);

const ChatProvider: React.FC<{ children: any }> = ({ children }) => {
    const [sessionId, setSessionId] = useState<string>('');
    const [status, setStatus] = useState<IChatStatus>(IChatStatus.PREPARING);

    const initSession = () => {
        setSessionId(Date.now().toString());
    };

    useEffect(() => {
        initSession();
    }, []);

    const chatContextValue = useMemo(() => {
        return {
            sessionId,
            status,
            setStatus,
        };
    }, [sessionId, status]);

    return <ChatContext.Provider value={chatContextValue}>{children}</ChatContext.Provider>;
};

export default ChatProvider;
