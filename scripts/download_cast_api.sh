

cast_api_macos_url=https://github.com/clariusdev/cast/releases/download/v12.2.4/cast-12.2.4-macos.arm64.zip
cast_api_linux_2204_url=https://github.com/clariusdev/cast/releases/download/v12.2.4/cast-12.2.4-linux.x86_64-gcc_ubuntu_22.04.zip
cast_api_linux_2404_url=https://github.com/clariusdev/cast/releases/download/v12.2.4/cast-12.2.4-linux.x86_64-gcc_ubuntu_24.04.zip
cast_api_windows_url=https://github.com/clariusdev/cast/releases/download/v12.2.4/cast-12.2.4-windows.x86_64.zip


echo "Setting up Cardiac Ultrasound Guidance environment..."

# detect OS
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*|MINGW32*|MSYS*|MINGW*) machine=Windows;;
    *)          machine="UNKNOWN:${unameOut}"
esac

# set up working directory
WORKING_DIR=$(pwd)
echo "Working directory: ${WORKING_DIR}"

# remove existing cast_api directory if exists
rm -rf cast_api
mkdir -p cast_api


echo "Detected OS: ${machine}"

# download and setup Cast API based on OS
if [ "${machine}" == "Mac" ]; then
    echo "Downloading Cast API for Mac..."
    curl -L -o cast_api_macos.zip ${cast_api_macos_url}
    unzip cast_api_macos.zip -d cast_api
    rm cast_api_macos.zip
    echo "Cast API for Mac downloaded and extracted in 'cast_api' directory."

    cp cast_api/libcast.dylib src/data_access/clarius/libcast.dylib
    cp cast_api/python310/pyclariuscast.so src/data_access/clarius/pyclariuscast.so
elif [ "${machine}" == "Linux" ]; then
    # detect Ubuntu version
    ubuntu_version=$(lsb_release -rs)
    echo "Detected Ubuntu version: ${ubuntu_version}"
    if [ "${ubuntu_version}" == "22.04" ]; then
        echo "Downloading Cast API for Ubuntu 22.04..."
        curl -L -o cast_api_linux_2204.zip ${cast_api_linux_2204_url}
        unzip cast_api_linux_2204.zip -d cast_api
        rm cast_api_linux_2204.zip
        echo "Cast API for Ubuntu 22.04 downloaded and extracted in 'cast_api' directory."

        cp cast_api/libcast.so src/data_access/clarius/libcast.so
        cp cast_api/python310/pyclariuscast.so src/data_access/clarius/pyclariuscast.so
    elif [ "${ubuntu_version}" == "24.04" ]; then
        echo "Downloading Cast API for Ubuntu 24.04..."
        curl -L -o cast_api_linux_2404.zip ${cast_api_linux_2404_url}
        unzip cast_api_linux_2404.zip -d cast_api
        rm cast_api_linux_2404.zip
        echo "Cast API for Ubuntu 24.04 downloaded and extracted in 'cast_api' directory."

        cp cast_api/libcast.so src/data_access/clarius/libcast.so
        cp cast_api/python310/pyclariuscast.so src/data_access/clarius/pyclariuscast.so
    elif [ "${machine}" == "Windows" ]; then
        echo "Downloading Cast API for Windows..."
        curl -L -o cast_api_windows.zip ${cast_api_windows_url}
        unzip cast_api_windows.zip -d cast_api
        rm cast_api_windows.zip
        echo "Cast API for Windows downloaded and extracted in 'cast_api' directory."

        cp cast_api/cast.dll src/data_access/clarius/cast.dll
        cp cast_api/python310/pyclariuscast.pyd src/data_access/clarius/pyclariuscast.pyd
    else
        echo "Unsupported Ubuntu version: ${ubuntu_version}. Please use Ubuntu 22.04 or 24.04."
        exit 1
    fi
else
    echo "Unsupported OS: ${machine}. This setup script supports only Mac, Windows and Linux."
    exit 1
fi



# clean up
echo "Setup completed."
rm -rf cast_api
