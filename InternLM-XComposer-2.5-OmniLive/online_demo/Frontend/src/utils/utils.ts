// Check if the device is mobile by navigator.useragent
export const isMobile = () => {
    let isMobile = false;
    if (navigator.userAgent) {
        const mobileKeyword = ['mobile', 'iphone', 'android'];
        mobileKeyword.forEach((keyword) => {
            if (navigator.userAgent.toLocaleLowerCase().includes(keyword)) {
                isMobile = true;
            }
        });
    }
    return isMobile;
};
