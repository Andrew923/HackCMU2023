CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230904000000_e20230904235959_p20230905022156_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-09-05T02:21:56.187Z   date_calibration_data_updated         2023-08-08T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-09-04T00:00:00.000Z   time_coverage_end         2023-09-04T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx���   �          @�>��������J�H���\C���>�����z������33C�y�                                    Bx���  
�          @���>�������?\)���C���>���������{\)C�p�                                    Bx���L  "          @�{�\���R�8Q���{C���\��\)���
�eG�C�4{                                    Bx����  �          @Å>����  �Fff��z�C���>����=q������RC��                                     Bx���  �          @Å?h����z��K�����C��?h����������ffC��                                    Bx��>  T          @\?z�H��z��E���  C�aH?z�H��
=��ff��C�ٚ                                    Bx��*�  �          @�=q?����=q�I�����\C���?����p������z�C�H�                                    Bx��9�  �          @��?�\��
=�A����C�Ǯ?�\��G���(���(�C���                                    Bx��H0  "          @�  ?�\���H�C33��Q�C�޸?�\�����\���\C���                                    Bx��V�  T          @�
=?����p��U���C�5�?����=q�����C��R                                    Bx��e|  "          @�  ?=p���\)�P����C�=q?=p�����   ���C��f                                    Bx��t"  T          @���?n{���R�R�\���C�J=?n{���H��\��=qC��3                                    Bx�؂�  
�          @��?����  �W��
�HC�*=?������
=q��33C�o\                                    Bx�ؑn  �          @�=q?���33�W����C�)?���Q������\)C�>�                                    Bx�ؠ  "          @��?�z���Q��P  �
{C���?�z������ff���C���                                    Bx�خ�  �          @�p�?}p���{�<(���C�N?}p���Q����=qC���                                    Bx�ؽ`  
�          @��>�  ��{�'
=��z�C���>�  ��p���
=�v�HC�g�                                    Bx���  �          @��
�.{���R�#�
��{C��
�.{����\)�m��C�\                                    Bx��ڬ  �          @��
?�R����7���\C��=?�R��G���p�����C�}q                                    Bx���R  �          @���?��R��p��R�\���C�n?��R��=q���G�C��\                                    Bx����  "          @�z�>�33���%��33C�f>�33��������_�C�ٚ                                    Bx���  T          @��?c�
��G��5��(�C��\?c�
���\��
=��(�C���                                    Bx��D  �          @�=q?^�R�����G
=�{C��R?^�R���׿�p���  C��                                    Bx��#�  
Z          @�{�8Q���p��{����C����8Q���=q����9��C���                                    Bx��2�            @��R�B�\����(���
=C�aH�B�\�������\�4Q�C���                                    Bx��A6  �          @�=q�����\)�G���p�C�1������Ϳ���O�
C���                                    Bx��O�  �          @�ff�Q���
=�(����C�}q�Q�����\)��z�C�f                                    Bx��^�  
�          @�ff���
�~{�:=q�C������
���������G�C��
                                    Bx��m(  �          @�33�k��mp��\���+z�C�:�k���p��p���ffC���                                    Bx��{�  �          @��\?L�������C33�  C�L�?L����z��Q�����C���                                    Bx�يt  "          @�ff�����Q��n{�*Q�C�\��������)����\C���                                    Bx�ٙ  �          @�G�=��
�����R�ɅC�w
=��
��zῃ�
�4��C�n                                    Bx�٧�  �          @�{�\(������z����C����\(���
=��z��O33C�q                                    Bx�ٶf  �          @��H�!G�����!G���z�C���!G���G���(���=qC�
=                                    Bx���  
�          @�ff���
�p���3�
��C��ü��
��=q��������C���                                    Bx��Ӳ  
�          @��>���o\)�@����
C���>������G���(�C��
                                    Bx���X  �          @��\>��H�y���4z���C��{>��H���R�����  C�!H                                    Bx����  T          @���?
=�{��(�����C�J=?
=��ff��\)��Q�C�˅                                    Bx����  �          @��
?���n{�:=q�  C��?����녿���(�C�g�                                    Bx��J  
�          @�?�z��\���:=q��\C��H?�z���녿��R��
=C���                                    Bx���  �          @�p�@��aG��.�R��C��3@����\��ff��{C�33                                    Bx��+�  T          @��H�u����2�\�(�C���u��
=��Q���C���                                    Bx��:<            @�ff�/\)��  �
=�ϮCk���/\)���R�����a�Cnc�                                    Bx��H�  �          @���'
=��Q��%����HCl��'
=��Q���
���\Co�R                                    Bx��W�  T          @����G��HQ�������CYǮ��G��`�׿����?
=C]                                    Bx��f.  
�          @�\)���\�C�
�����_�CW8R���\�Q녿����HCY33                                    Bx��t�  
          @�p���=q���Ǯ���COW
��=q�+��p��� z�CRT{                                    Bx�ڃz  !          @��R��녿�z῎{�<Q�CF&f��녿�׿+���=qCHc�                                    Bx�ڒ   �          @�\)������ÿaG���CB!H������R������CC��                                    Bx�ڠ�  "          @�
=��p��������@  CBxR��p����ÿE��{CD�                                    Bx�گl  "          @��R��{���Ϳ�\)�?33CH�{��{�z�!G����CJ                                    Bx�ھ  
Z          @�ff���H�z�G��Q�CM�
���H�(��8Q����CN�q                                    Bx��̸  T          @�  ��z���׿fff��HCL�\��z��=q���R�P  CN@                                     Bx���^  �          @�  ��(���
�\(��z�CMG���(���;���0  CN�)                                    Bx���  
�          @����=q��G�����8(�CA�q��=q���R�8Q����CDY�                                    Bx����  �          @�\)����z�xQ��#�CM�\����\)��p��u�CO(�                                    Bx��P  
(          @�\)����DzῨ���`��CWn����S33����\)CYn                                    Bx���  T          @�����7��s33���CS�=���@�׾L���CT�\                                    Bx��$�  
Z          @�{���\�+���\)�?
=CRaH���\�8Q��
=��CT(�                                    Bx��3B  
�          @�z�����I��������{CY������\�ͿB�\�{C\W
                                    Bx��A�  
�          @��H��
=�E���\�0��CX33��
=�P  �aG��ffCY��                                    Bx��P�  �          @�G�����=p��O\)��HCV������Dz�L�Ϳ   CW��                                    Bx��_4  
�          @�G������.�R��=q�k\)CT�������>�R�(���z�CV��                                    Bx��m�  T          @�����p��   ��
���CR����p��<(���\)�pz�CW.                                    Bx��|�  "          @��H��{�'�� ����33CT���{�B�\����_�
CW��                                    Bx�ۋ&  
�          @�p��qG��X�ÿ!G����
C]�q�qG��\(�>u@,��C^h�                                    Bx�ۙ�  "          @�z��h���g
=<#�
>\)C`���h���_\)?k�A%��C_�                                     Bx�ۨr  
�          @�{�`  �U?Y��A�RC_�f�`  �AG�?�z�A�(�C\�                                    Bx�۷  �          @�ff�Z�H�I����=q�S33C^���Z�H�G�?
=q@�(�C^^�                                    Bx��ž  T          @�z��<(��7
=?�\)A���C`5��<(���H@33A�(�C[xR                                    Bx���d  "          @���u�,(�������RCV�3�u�.{>k�@0��CWE                                    Bx���
  T          @�������#33�333� (�CR�{����(�ý#�
����CS�                                    Bx���  T          @���  ��Q�O\)�!G�CI����  ����Q����CKff                                    Bx�� V  �          @�z���  ����z��\Q�CN���  �ff�
=��(�CP��                                    Bx���  �          @�p������
=q�����N�\CN����������\��
=CP�q                                    Bx���  T          @�{����(����Q�CP�R����*=q��\��(�CS
                                    Bx��,H  �          @�
=��
=�=q��Q��RffCPQ���
=�(Q�����HCR}q                                    Bx��:�  T          @�
=���\�)����(��XQ�CS}q���\�8Q�   ��  CU��                                    Bx��I�  �          @��
����#�
��G��5p�CR������.�R���
�b�\CTQ�                                    Bx��X:  
�          @��\����5�������CVB�����8��>��?�CV��                                    Bx��f�  
�          @�Q����R�!녿s33�/33CR�q���R�,(���=q�C�
CT��                                    Bx��u�  "          @��������\�p���-p�CO�������;��
�j�HCQ^�                                    Bx�܄,  "          @�����녿\(���
COE�����u�+�CP�R                                    Bx�ܒ�  
�          @��\���R�z�8Q��{CO����R����Q쿆ffCP�
                                    Bx�ܡx  T          @�������{�xQ��1CN����������Q����
CP�                                    Bx�ܰ  
(          @�Q���p�����
�>ffCME��p��녾�ff���RCOO\                                    Bx�ܾ�  
�          @��\��z��G��p���4��CP����z������
�uCRu�                                    Bx���j  T          @��R����p��aG��-G�CP�\����
=��\)�X��CR5�                                    Bx���  z          @�\)��G����@  �=qCQ�3��G��(�����CS(�                                    Bx���  `          @���\)����Y���%p�CS  �\)�"�\�B�\�ffCTp�                                    Bx���\  �          @����|(���\�k��5�CR0��|(���;�z��aG�CS��                                    Bx��  
�          @��R�x�����{�\Q�CS\�x���#33������CUE                                    Bx���  
�          @�{�vff��׿����{CRff�vff�"�\�=p���RCUs3                                    Bx��%N  T          @�(��l(��(�ÿxQ��@z�CW�
�l(��333�u�<��CY5�                                    Bx��3�  
�          @�33�_\)�1녿���R=qCZ�\�_\)�=p���=q�Y��C\J=                                    Bx��B�  
�          @���e�!녿��\��p�CW(��e�1G��\)���CY��                                    Bx��Q@  
�          @�(���G���׿�p��w33CL����G���ÿ0���	p�CO�f                                    Bx��_�  
�          @��
�����
=�Q��#33CM
����zᾏ\)�\��CN                                    Bx��n�  �          @�33�vff��ÿO\)�"�RCS�{�vff�!G������\)CU33                                    Bx��}2  T          @�\)�qG��.�R�!G���p�CW�3�qG��333=�?�p�CX�H                                    Bx�݋�  
�          @��H�XQ��@  ��
=��G�C]�)�XQ��U�Q���C`��                                    Bx�ݚ~  	�          @���N�R�>�R�33��Q�C^���N�R�Z�H��Q��b=qCb��                                    Bx�ݩ$  �          @��=p��S33�Q�����Cd
�=p��p  ���[�
Cg��                                    Bx�ݷ�  �          @�(��+��k���
=��{Ci�3�+���  �!G���Q�Cl(�                                    Bx���p  
�          @��
��(��z�H�ff���HCs^���(���33�z�H�8��Cu�H                                    Bx���  z          @���	�����H��  ��z�CrJ=�	�����H��33��p�Cs��                                    Bx���  �          @�G��\���{��
=���
CT� �\���%���\�Tz�CX޸                                    Bx���b  �          @���� ���G������=qCg+�� ���\(��=p��p�Ci�                                    Bx��  T          @�33��=q�e��6ff�33Cv)��=q���R���
��Q�Cyc�                                    Bx���  �          @�(��У��o\)�(����CvaH�У���녿��
��
=Cy=q                                    Bx��T  
Z          @�{���|(���
��p�Ct녿���p���33�X��CwY�                                    Bx��,�  �          @��R�*=q�c33�G��Ù�Ci#��*=q�~{�z�H�9G�Cl�                                    Bx��;�  �          @����o\)�E������yC[u��o\)�U�������C]��                                    Bx��JF  "          @�=q�h���Y���h���%C_�h���aG�=L��?��C`�                                    Bx��X�  �          @�G��\(��e��:�H�ffCb&f�\(��h��>�=q@FffCb��                                    Bx��g�  
�          @����R�\�l�Ϳ&ff����CdL��R�\�o\)>\@��Cd�
                                    Bx��v8  "          @���=p��x�ÿTz��=qCh�R�=p��~{>�=q@E�CiE                                    Bx�ބ�  T          @�\)�Q��i�����
�l(�Cd��Q��fff?333A=qCc��                                    Bx�ޓ�  "          @��� ���K��H���z�Cm��� ���z�H�	����  Cr�{                                    Bx�ޢ*  
�          @�=q���H�N{�>{�  Cn�ÿ��H�z=q��(����Csn                                    Bx�ް�  "          @����
=�@���R�\�2�Ct�
��
=�s�
�����CyaH                                    Bx�޿v  "          @��\�z��Tz��5����Cn\�z��~{��ff����Crz�                                    Bx���  
�          @��׿����Tz��L(��(�Cz&f�������\�	������C}�)                                    Bx����  �          @�  ��33�W��I���%�HC{8R��33�����У�C~k�                                    Bx���h  T          @��׿\�\���:�H�  Cv.�\���
������Q�Cy�q                                    Bx���  
�          @��������a��:�H���Cw��������ff������=qCz�3                                    Bx���  �          @�G���=q�i���3�
�z�Cy�3��=q��G���
=��=qC|�q                                    Bx��Z  �          @�녿�p��p���&ff���Cx�=��p����H������{C{.                                    Bx��&   �          @�G���Q��z�H�{��=qC}0���Q����R���\�vffC�                                    Bx��4�  �          @�=q��ff�u�{��=qCx\��ff��(�����x��Cz�=                                    Bx��CL  
Z          @�G���  �j=q�!G���
=Cth���  ��
=��33���
Cws3                                    Bx��Q�  T          @�Q�����fff� �����RCr�{���������33���RCv\                                    Bx��`�  T          @��ÿ�G��|���{��\)Cy{��G���p���G��DQ�C{)                                    Bx��o>  	�          @�G���G��|(��\)���
Cx�R��G��������H��C{�                                    Bx��}�  
Z          @�ff�{�y���˅��{Cm���{��{��
=��\)Cos3                                    Bx�ߌ�  
�          @��R�O\)�\�Ϳ������Cb�)�O\)�mp���33���HCd�)                                    Bx�ߛ0  
(          @�Q��P  �e��G��h��Cc�{�P  �s33�B�\�Q�Cep�                                    Bx�ߩ�  T          @���XQ��e���z��T(�Cb���XQ��p�׽�Q쿈��Cd{                                    Bx�߸|  �          @�  �=q�\)�=p����Cnٚ�=q����>�
=@���Co#�                                    Bx���"  T          @����1G��xQ쿁G��?
=Cj��1G���Q�>\)?�CkaH                                    Bx����  �          @�����R�|�Ϳ��H����Cmٚ��R���R��=q�H��Con                                    Bx���n  �          @��H��R�\)�����Y�Cn.��R����=u?+�Co+�                                    Bx���  "          @���
��녿��\�Ip�Cs\��
��>B�\@ffCs��                                    Bx���  T          @��
�
=�|�Ϳ���d��Cq���
=���=#�
?�Cr�
                                    Bx��`  
(          @�p��*�H�e��ff��  CiW
�*�H�s�
�W
=�(��Cj��                                    Bx��  �          @�p��   �n�R���
�\)Cl��   �|(�������Cm��                                    Bx��-�  �          @��
�:=q�aG��������
Cfs3�:=q�u����H����Ch                                    Bx��<R  
�          @��H�G��Y������G�Ccs3�G��j=q�\��  Ce��                                    Bx��J�  
�          @����:�H�c�
���R�p��Cf���:�H�p�׾����=qCh.                                    Bx��Y�  �          @�33�L���Z�H���R�l��Cb�f�L���h�þB�\���Cd��                                    Bx��hD  "          @����b�\�G
=����~�\C]G��b�\�W
=�\��{C_�=                                    Bx��v�  "          @�G��p���6ff���
��\)CY+��p���O\)�^�R�
=C\��                                    Bx����  
�          @�=q�u�1녿����\)CW���u�K��fff�#33C[��                                    Bx���6  
�          @����xQ��8Q���
��ffCX���xQ��L�Ϳ�R��Q�C[��                                    Bx���  
�          @�G��qG��B�\�����  CZ�f�qG��Tz��G����RC]T{                                    Bx�ూ  T          @�p��a��*�H�ff��G�CY+��a��J�H��(��g\)C]�q                                    Bx���(  �          @���Y���2�\��\�ƸRC[^��Y���QG���\)�V�\C_ٚ                                    Bx����  �          @���Z=q�C33��  ��p�C]���Z=q�[��E����Ca.                                    Bx���t  H          @�(��U��O\)��p����HC`:��U��a녾�ff����Cb��                                    Bx���  .          @�33�Q��Mp��Ǯ��ffC`Q��Q��a녿
=q��p�Cc
=                                    Bx����  �          @�(��P  �J�H��G�����C`8R�P  �c33�=p��(�Ccu�                                    Bx��	f  T          @���J=q�W
=�����Cb��J=q�p�׿L����
Ce�3                                    Bx��  �          @�G��C33�QG�������Cc  �C33�g
=�
=���
Ce�\                                    Bx��&�  "          @�G��C33�L�Ϳ�=q���Cb^��C33�g
=�J=q�z�Ce�R                                    Bx��5X  �          @�
=�)���C�
�Q���Ce!H�)���hQ쿬����=qCi�f                                    Bx��C�  �          @�{����J�H��R���
Ci����p�׿�z����RCm��                                    Bx��R�  
�          @�
=�\)�W
=�(����HCis3�\)�w
=��=q�S
=Cm.                                    Bx��aJ  
Z          @�Q����Z�H�p���ffCj�����{���=q�R�HCn8R                                    Bx��o�  "          @����-p��g���=q����Ci��-p��z�H�����CkT{                                    Bx��~�  �          @���&ff�U�	���ׅCh��&ff�u����J�RCk޸                                    Bx��<  �          @�Q��!��c�
�޸R����Cj���!��z�H�z����HCm
                                    Bx���  T          @�Q��  �l(����R���Cn���  ���
�G���HCq^�                                    Bx�᪈  �          @��\�)���hQ�˅��{Ci���)���|(�������Cl�                                    Bx��.  �          @��H�5�g
=��  ��
=CgǮ�5�x�þ����~{Ci��                                    Bx����  �          @�33�#�
�p  ��Q���p�Ck��#�
���H�������Cm�                                    Bx���z  �          @�33�=p��dzΌ����p�CfY��=p��u��z��^�RChk�                                    Bx���   �          @��\�H���^{��33�\��Cc���H���i�����ǮCeG�                                    Bx����  "          @��
�L���e��.{��Cd5��L���g�>�ff@���Cd��                                    Bx��l  T          @��
�J=q�c�
�^�R�&�\CdxR�J=q�j=q>��@G
=Ce8R                                    Bx��  �          @���H���e�^�R�%��Cd�=�H���k�>�\)@P��Ce�                                    Bx���  "          @�(��Vff�W������Qp�Ca8R�Vff�c33�#�
��Cb�                                    Bx��.^  �          @��
�A��k��s33�4��Cf�\�A��s33>k�@.�RCgk�                                    Bx��=  
�          @����(���33���H���Co8R�(�����?L��A=qCn��                                    Bx��K�  �          @����H���X�ÿ���T��Cc.�H���c�
<#�
>#�
Cd�
                                    Bx��ZP  "          @�Q��>{�]p����
�z�HCeQ��>{�k��\)���HCg�                                    Bx��h�  �          @�G��$z��vff��\)�X��ClG��$z���Q�>\)?�
=CmQ�                                    Bx��w�            @�  �#33�}p������z�CmG��#33�{�?8Q�A
�HCm�                                    Bx��B  .          @����'
=�|�Ϳ   ��G�Cl���'
=�z=q?B�\A�RClG�                                    Bx���  �          @�Q��!G����׾�Q����
Cmٚ�!G��{�?h��A1�CmE                                    Bx�⣎  �          @�  �&ff�|�;\���Cl���&ff�w�?aG�A*�\Cl�                                    Bx��4  z          @�p��(���s�
��33���
CkE�(���n{?\(�A*�RCj��                                    Bx����  T          @����0  �X�ÿ�33�lz�Cg��0  �e����
���RCh�                                    Bx��π  .          @�33�,(��e�@  ���Ci8R�,(��i��>�
=@���Ci�H                                    Bx���&  
�          @���!G��y���#�
��Cm(��!G��mp�?�Q�An=qCk޸                                    Bx����  �          @��H�(���p  ��=q�X��Cj�)�(���h��?n{A:�HCj�                                    Bx���r  
Z          @��(��\)���
�k�Cn���(��r�\?�  Aw�
Cm=q                                    Bx��
            @�(��"�\�u�����\)Cl^��"�\�qG�?G�A�Cl�                                    Bx���  �          @�z��4z��\(����R�y�Cf���4z��j=q���
���\Chc�                                    Bx��'d  �          @����7
=�W������Ce���7
=�hQ�u�@��Cg�                                    Bx��6
  
Z          @����0  �dz῎{�]��Chc��0  �o\)=�Q�?���Ci��                                    Bx��D�  T          @��Q�����>.{@z�Cr�f�Q��xQ�?��RA���CqJ=                                    Bx��SV  �          @�33�3�
�c33�:�H��RCg���3�
�fff>�ff@��HCg�R                                    Bx��a�  �          @���;��^�R�@  ���Ce���;��a�>��@��Cf\)                                   Bx��p�  �          @�G��.�R�dz�+��(�Ch���.�R�e?�\@�  Ch��                                   Bx��H  �          @��
�.�R�i���0���	��Ci.�.�R�k�?�@љ�CiaH                                    Bx���  H          @�  �,���w
=�&ff���
Ck
�,���w�?!G�@��Ck)                                    Bx�㜔  `          @�Q��'��|(��z��ᙚClk��'��z�H?8Q�A�
ClB�                                    Bx��:  "          @��������33��������Co�������Q�?uA9�Co
                                    Bx���  �          @�Q���\���;��R�n�RCq(���\����?��AN{Cph�                                    Bx��Ȇ  �          @�\)�\)�\)��G����
Cn��\)�z=q?aG�A,Q�Cm�{                                    Bx���,  T          @��R��33���R    �L��C|����33���R?���A��HC{��                                    Bx����  �          @�{������
=��Q����Cu.������33?��AMp�Ct�{                                    Bx���x            @�p��	���������J�HCrc��	���~{?���A\(�Cq��                                    Bx��  �          @�p��#�
�x�þ�p����HCl�f�#�
�r�\?k�A6�RCl�                                    Bx���  T          @�p��(���u�Ǯ����Ck���(���p  ?fffA0��Cj��                                    Bx�� j  �          @��R�*=q�vff��\��\)Ck\)�*=q�s�
?J=qA{Ck�                                    Bx��/  T          @�\)�)���xQ�����
Ck�f�)���u�?J=qA�CkW
                                    Bx��=�  
�          @�\)�(���w����H��Q�Ck�(���tz�?Q�A�CkaH                                    Bx��L\  T          @�{�#33�xQ����{Cl���#33�tz�?Tz�A$(�ClO\                                    Bx��[  
�          @�(����z=q�!G���G�Cn�{���y��?5A��Cn�q                                    Bx��i�  �          @��\�p��q녿@  �\)Cl��p��tz�?��@���Cm.                                    Bx��xN  T          @�33�&ff�p  �����(�CkJ=�&ff�n{?:�HA�RCk�                                    Bx���  
(          @��H�0  �fff�(����RCh�)�0  �fff?�R@�33Ch��                                    Bx�䕚  T          @��\�.{�hQ��ff���RCi#��.{�e�?J=qA33Ch�3                                    Bx��@  T          @��
�5��g���z��e�Cg�q�5��`��?p��A=G�Cg�                                    Bx���  
�          @����'��i�������
CjT{�'��e?J=qA!��Ci�f                                    Bx����  �          @����,���e��ff����Ci
=�,���a�?G�A\)Ch��                                    Bx���2  T          @�(��Fff�Z=q�����n{Cc��Fff�S�
?^�RA-p�Cb�                                    Bx����  �          @�33�8���c�
�\����Cf��8���^�R?W
=A(��CfJ=                                    Bx���~  �          @�33�$z��p  �\)��=qCk���$z��n{?=p�AQ�CkJ=                                    Bx���$  �          @�z��#33�u������Cl\)�#33�s33?:�HA��Cl5�                                    Bx��
�  T          @���{�w
=������Cm^��{�r�\?aG�A/33Cl޸                                    Bx��p  �          @�z��+��qG������p  Cj���+��i��?�  AF�HCi��                                    Bx��(  
�          @�z��!��w�����Mp�Cl޸�!��n�R?�=qAW33Ck޸                                    Bx��6�  
�          @�ff�����׾.{�Q�Cn޸���u�?��HAn�RCm�                                    Bx��Eb  .          @��R� ���~�R�u�8Q�Cm�q� ���p��?��A���Cl@                                     Bx��T  �          @�{�&ff�z=q>�?˅Clk��&ff�hQ�?��HA�\)Cjk�                                    Bx��b�  �          @�  �1��u>��@H��Cj
�1��`��?ǮA��
Cg��                                    Bx��qT  T          @���1G��s�
?�@��HCj
=�1G��Y��?�ffA�
=Cf޸                                    Bx���  
Z          @��
�8���c33>�@�Q�Cf�f�8���K�?�z�A��Cc�R                                    Bx�厠  
�          @����Dz��R�\>�@���Cb�R�Dz��:�H?˅A�  C_�{                                    Bx��F  
�          @�z��L(��Tz�?�@У�Cb(��L(��;�?��A�=qC^�)                                    Bx���  
�          @��H�B�\�X��?
=q@�  Cd
�B�\�?\)?�
=A�{C`��                                    Bx�庒  �          @�33�G��U�u�<��Cb���G��Mp�?p��A=Ca�H                                    Bx���8  �          @��R�@  �J�H�.{��\Cb�)�@  �Mp�>�@�Q�Cb�R                                    Bx����  
�          @���5�4z��R�C`�\�5�7
=>���@��Ca5�                                    Bx���  �          @�p��6ff�?\)�������CbW
�6ff�:=q?B�\A)�Ca��                                    Bx���*  T          @�p��'
=�O\)�#�
�(�Cg#��'
=�C33?��As�Cep�                                    Bx���  �          @�  �&ff�Vff����`��Ch:��&ff�N�R?n{AJ�RCg33                                    Bx��v  "          @��R�*=q�N�R��\)�q�Cf��*=q�G�?aG�A@��Ce��                                    Bx��!  �          @�
=��R�`�׿#�
�
=Cm}q��R�`  ?#�
A  Cmz�                                    Bx��/�  
�          @����33�\�Ϳ}p��^�\Co=q�33�e�>�z�@~{Cp&f                                    Bx��>h  
�          @��H�3�
�HQ쿀  �W33Cd��3�
�Q�>#�
@(�Cen                                    Bx��M  �          @����9���<�Ϳp���Lz�Ca���9���Fff>.{@�Cb�f                                    Bx��[�  
(          @����>{�<(��+��{C`���>{�?\)>�
=@�(�Ca0�                                    Bx��jZ  
�          @�(��A��HQ쾊=q�`��Ca���A��AG�?\(�A6{C`޸                                    Bx��y   
�          @���>�R�C33�.{��
Ca��>�R�:=q?n{AHQ�C`Y�                                    Bx�懦  
�          @�33�E��B�\>�p�@��HC`�f�E��-p�?�
=A�p�C]\)                                    Bx��L  T          @���HQ��?\)>��@XQ�C_���HQ��-p�?��A�ffC\��                                    Bx���  �          @���O\)�%�?��RA�ffCZxR�O\)��\)@�
A��
CR                                      Bx�泘  �          @�z��J�H�"�\?�33A��RCZ���J�H���
@��B�HCQL�                                    Bx���>  �          @���@  ��?��A�  C[��@  ����@(Q�B{CP                                    Bx����  �          @�=q�HQ��*=q?���A���C\O\�HQ��   @(�A�(�CT��                                    Bx��ߊ  �          @�=q�S33�z�?�ffA��RCW0��S33��{@�\A�CM�q                                    Bx���0  	�          @|���AG�����p���G�CZ33�AG��ff?�\@�33CY��                                    Bx����  �          @j=q�(Q����\)�Q�C]�3�(Q��=q>���@�=qC^u�                                    Bx��|  T          @`  �,���
�H��G���
=CZǮ�,����>\@�ffCZ�                                    Bx��"  �          @\���(���
�H��ff��Q�C[c��(���(�>�p�@�(�C[�
                                    Bx��(�  T          @S�
�{�z�.{�?
=C[�3�{��=�@�
C]\)                                    Bx��7n  
Z          @~�R�
=q��/\)�N�HC� �
=q�<�Ϳ������C���                                    Bx��F  T          @�33�\)�\)�i���f��C�q�\)�Z�H�%����C�Z�                                    Bx��T�  "          @����{�z��`  �`�C��;�{�\(��=q��RC�(�                                    Bx��c`  �          @�(����R�HQ��ff��RCxuÿ��R�qG�����j�HC{޸                                    Bx��r  T          @�\)�B�\�<(��;��/�
C�{�B�\�s�
������C�c�                                    Bx�瀬  T          @��R�xQ��7��P  �;�C{J=�xQ��w���(���  C޸                                    Bx��R  
�          @�\)���6ff�O\)�9�RCw�ÿ��vff���H��(�C}
                                    Bx���  �          @��׿����S33�0  ���Cx33�������\������RC|{                                    Bx�笞  �          @������H�\���%�
p�CwͿ��H��������ep�Cz�f                                    Bx��D  "          @�G���(��_\)�   ���Cw&f��(�������
�Q��Cz�\                                    Bx����  �          @�  �����W
=�"�\�	��Ct�q������=q��{�eG�Cx�f                                    Bx��ؐ  
Z          @�  ���!��Dz��,�RCf�����`  ��z��ɅCoL�                                    Bx���6  �          @��R�����Z�H�#�
���Cz��������(������eC}�=                                    Bx����  �          @��׿�
=�AG��@  �'�RCt�3��
=�z�H��z���G�Cz                                      Bx���  �          @������H�@���1��33Cm����H�u����H��Cr��                                    Bx��(  "          @�G�����P  �8Q��Cx�������H������C|J=                                    Bx��!�  "          @��׿����H���8Q��ffCuE�����\)��  ��p�Cz
                                    Bx��0t  
Z          @��R��ff�C33�5��RCs\��ff�y�����R��(�CxJ=                                    Bx��?  "          @���33�H���(�����CrG���33�z=q���\���HCw{                                    Bx��M�  	�          @��R���A������(�Cg�����hQ�fff�:�HCl�
                                    Bx��\f  �          @�����G��
�H�홚Cj������mp��W
=�0  CoY�                                    Bx��k  �          @�
=�Q��Tz��  ���HCjaH�Q��o\)��Q���(�Cm}q                                    Bx��y�  {          @��
�7
=�>�R��\)����Cb��7
=�QG��\)��\)Cd�
                                    Bx��X  �          @�33�0  �:�H�����ffCb���0  �N�R�.{�
=Ce��                                    Bx���  
�          @����H�A녿���ffCgW
��H�_\)���ᙚCk:�                                    Bx�襤  "          @���*�H�'�����=qC`u��*�H�N�R�u�N�HCfk�                                    Bx��J  T          @���(��o\)��33�n{Co��(��y��>�Q�@�(�Cp��                                    Bx����  T          @��
�*�H�Q녿s33�J{Cf�\�*�H�X��>\@�\)Cg                                    Bx��і  "          @���%�Vff��  ���ChJ=�%�j�H���Ϳ�G�Cj�                                    Bx���<  
Z          @�  �#33�P  ��  ���Cg�H�#33�j�H��p���Q�Ck:�                                    Bx����  �          @�{�{�P  �z����
Ck�H�{�r�\�(���	G�Co�f                                    Bx����  "          @������Fff�   �
p�Cn�)����tz῏\)�k�Cs��                                    Bx��.  "          @�(��
=�L���33��RCl���
=�p  �(���
�RCp��                                    Bx���  �          @��{�J�H�������Cj���{�p  �@  �
=CoW
                                    Bx��)z  �          @��
���O\)����{Cpk����w��Y���4Q�Ct�
                                    Bx��8   �          @�������dzῃ�
�[�Cm�
����l(�>�(�@�p�Cns3                                    Bx��F�  
�          @���#33�_\)�L���'�
Ci�{�#33�a�?�RA ��Cj�                                    Bx��Ul  "          @���$z��\�ͿaG��8  Cic��$z��aG�?�@�  Ci�                                    Bx��d  �          @��8���Q녿!G��33Cd�f�8���QG�?333AG�Cd��                                    Bx��r�  �          @����.{�Q녿   ���Cf@ �.{�Mp�?Q�A0(�Ce��                                    Bx��^  "          @��
�(���Vff�\(��6=qCg�R�(���Z�H?�@��ChJ=                                    Bx��  "          @���%��Y���.{���ChǮ�%��X��?333A�Ch�q                                    Bx�鞪  T          @����7
=�G���G����
Cc�{�7
=�C33?Q�A2�\Cbٚ                                    Bx��P  �          @����H���0  �k��G�C]B��H���9��>k�@FffC^��                                    Bx���  
          @��=p��2�\�xQ��UC_J=�=p��<(�>W
=@6ffC`�
                                    Bx��ʜ  _          @�p��W�������CV���W��
=>�G�@���CV�                                    Bx���B  �          @�z��HQ��'��#�
�
=C[�H�HQ��*=q>�G�@�(�C\Q�                                    Bx����  T          @���P��� �׿�\���
CY��P���   ?�@�p�CY�                                    Bx����  
�          @�p��U��R�u�S�
CX��U��?B�\A'�
CWQ�                                    Bx��4  T          @��H�XQ��
=q����{CT�)�XQ��(�>\@��\CT��                                    Bx���  
�          @����fff��=q�n{�U�CK�3�fff���þk��N{CN��                                    Bx��"�  
�          @|���g
=��Q�}p��i�CF@ �g
=���R��(����HCJ\)                                    Bx��1&  �          @w��k��k��333�'�CB)�k����׾�=q���\CE
=                                    Bx��?�  T          @|(��k����E��4  CE�
�k����׾u�^�RCH�                                     Bx��Nr  T          @u��c33��G�����z�RCC�\�c33�������\)CH�)                                    Bx��]  �          @l(��^�R��녿�33��p�C:� �^�R�W
=�c�
�_\)CA�=                                    Bx��k�  "          @k��Z�H��\��Q����\C<}q�Z�H�s33�c�
�ap�CCxR                                    Bx��zd  "          @`  �8�ÿ�Q쿣�
��Q�CN���8�ÿ�=q�
=�(�CT^�                                    Bx��
  �          @dz��8Q쿘Q��Q���  CJz��8Q��\�������\CS�
                                    Bx�ꗰ  �          @g��;���  ������ffCO��;���
=�(���+33CU^�                                    Bx��V  �          @P  ��ÿ��Ϳ�(��ۮC\�)�����\����-G�Cb�f                                    Bx���  T          @E��G���G����
��ffCY���G��녾k���z�C]�)                                    Bx��â  "          @C�
�����ÿ����p�C[�����Q쾞�R��G�C`Q�                                    Bx���H  �          @J�H��
��z�c�
����C[� ��
�ff�#�
�.{C^B�                                    Bx����  �          @U��)������  ��(�CT.�)����
=�u��{CX{                                    Bx���  "          @XQ��/\)��(��O\)�a�CT\�/\)��33�L�ͿJ=qCV�R                                    Bx���:  T          @dz��@�׿����=q��{CO\�@�׿���Q����
CS��                                    Bx���  T          @X���>�R��zΉ���{CI:��>�R��  �����CN��                                    Bx���  "          @X���7
=��=q���R��ffCL��7
=��(��
=�!p�CS\                                    Bx��*,  "          @P���#�
��(������\)CU�\�#�
���R��  ��(�CY�)                                    Bx��8�  "          @U�Q���ÿfff�{�C]���Q��z�=L��?p��C`G�                                    Bx��Gx            @W����	���Y���ip�C]�{����
=�G�?�
=C_�{                                    Bx��V  �          @`  �C33�����aG��jffCMk��C33��
=�W
=�\(�CP��                                    Bx��d�  �          @e��:=q��Q쿴z���G�CN^��:=q��33�0���3\)CU�                                    Bx��sj  T          @p���E��Ǯ��{��  CN���E����R�����
CTǮ                                    Bx��  
�          @g
=�A녿�
=�n{�n�HCQ��A녿�����
=CTE                                    Bx�됶  
�          @l(��>�R��33��G��~�HCT� �>�R�	������CW�                                     Bx��\  T          @k��:�H��
�aG��]��CW+��:�H�\)=L��?G�CYu�                                    Bx��  �          @j�H�G���ff�333�.�HCQ�3�G���=�G�?�  CS��                                    Bx�뼨  
�          @p  �Dz���R�=p��7�CT�3�Dz���>��@�
CV�)                                    Bx���N  
O          @p���=p���p���i�CW&f�=p���\<#�
>.{CY�                                    Bx����  �          @n�R�$z�� �׿s33�l  C`\)�$z��+�>B�\@8Q�Cb0�                                    Bx���  �          @qG��&ff�%�J=q�A�C`�f�&ff�+�>�Q�@�=qCa��                                    Bx���@  �          @q��(Q��(�ÿ
=���Ca��(Q��)��?\)A	G�Ca33                                    Bx���  �          @l���.�R�\)�8Q��8(�C[c��.�R�>�\)@���C\��                                    Bx���  "          @s�
��R�A녿�\����Ci����R�=p�?O\)AE�Ci                                      Bx��#2  T          @l������%������CcG�����2�\>��@G�Ce\)                                    Bx��1�  "          @c33�
�H�{���H��33Ca���
�H�.�R��R�!Cg�\                                    Bx��@~  "          @e������)��������
Ci�������?\)�����HCl�f                                    Bx��O$  {          @l(����2�\�����  Ch�R���>�R>B�\@<(�Cj��                                    Bx��]�  _          @i�����HQ��\���Co�R���C33?Y��AX��Co\)                                    Bx��lp  
�          @k����7
=�W
=�UG�Cin���<(�>�G�@�
=CjB�                                    Bx��{  
�          @g
=��{�J�H����G�Cs{��{�G�?L��AL��Cr                                    Bx�쉼  �          @j=q�{�/\)�Y���Y�Cg��{�6ff>Ǯ@���Ch{                                    Bx��b  �          @n{�8Q���R��Q���z�CV�)�8Q���
�u�j=qCZ�                                    Bx��  
Z          @r�\�8Q��(��������CYE�8Q���ͽL�ͿG�C\W
                                    Bx�쵮  
�          @l������*�H�^�R�Z�\Cd:�����2�\>�33@�{Cep�                                    Bx���T  T          @k��33�7
=��ff��
=CjG��33�A�>�=q@�p�Ck�H                                    Bx����  �          @mp����,�Ϳ=p��;
=Cd�����0��>�@��CeQ�                                    Bx���  {          @\)�R�\��\��\)��G�CP\)�R�\�(�����{CU�                                    Bx���F  _          @~�R�@  �{��\)��CXu��@  �%��z���\)C\��                                    Bx����  "          @~�R�L(���=q��p����CQٚ�L(���\�����\)CW��                                    Bx���  
�          @\)�P  �˅�ٙ����CN
=�P  �
=q�Y���D��CU�f                                    Bx��8  
�          @\)�S�
������
��(�CIT{�S�
��
=��ff�t(�CR:�                                    Bx��*�  T          @\)�J=q��{���R��{CKJ=�J=q�����H����CUc�                                    Bx��9�  �          @Dz��%�?E����\��C#T{�%�>����p���  C0��                                    Bx��H*  �          @G��(Q�?5��
=��  C$�)�(Q�<��
�������\C3�                                     Bx��V�  �          @HQ��/\)�#�
��p����C433�/\)�&ff��=q��ffCAQ�                                    Bx��ev  T          @H���5>8Q쿨����33C0xR�5��녿��\��z�C<J=                                    Bx��t  �          @G
=�/\)���H��=q��
=C>!H�/\)���\�}p����
CHh�                                    Bx���  �          @<(��%��.{��(�����CB�3�%���녿O\)��CK�
                                    Bx��h  
�          @AG��-p������������C:E�-p��Tzῃ�
��Q�CE                                    Bx���  
�          @>�R�2�\�Ǯ�n{��(�C<  �2�\�B�\�+��O�CC33                                    Bx����  �          @HQ��8Q쿂�\�&ff�@��CGz��8Q쿙���\)�'
=CJ�H                                    Bx���Z  �          @N{�:=q���ͿB�\�Z�HCH���:=q���þL���g�CL^�                                    Bx���   T          @J�H�6ff����5�T  CH)�6ff��  �B�\�[�CK�3                                    Bx��ڦ  
�          @G
=�333�����ff����C7)�333�5�������CB#�                                    Bx���L  �          @H���.{>�  ��  ���
C.�
�.{��G����H�ޏ\C=
                                    Bx����  "          @@  �#�
>��R��G���ffC-{�#�
�\���R��C<}q                                    Bx���  {          @@  �,(�=�G���ff�Ώ\C1���,(���׿�(���Q�C=��                                    Bx��>  
�          @Fff�4z�>W
=���R��
=C/�f�4zᾳ33���H���C;�                                    Bx��#�  T          @E��1G�>k����
���
C/5��1G���Q쿠  ���C;O\                                    Bx��2�  "          @E�*=q?\)���H��z�C(��*=q�\)�Ǯ���HC7
                                    Bx��A0  T          @Fff�1�>8Q쿨�����HC0W
�1녾�
=��G����C<��                                    Bx��O�  T          @J�H�9���aG����R����C8@ �9���=p����\��  CBW
                                    Bx��^|  �          @O\)�@�׿�������RC=�\�@�׿n{�333�I��CE#�                                    Bx��m"  T          @O\)�:=q�W
=�������CD+��:=q���R�(��/33CK#�                                    Bx��{�  �          @J=q�3�
�}p����
��Q�CGh��3�
��������CM�\                                    Bx��n  
Z          @H���.�R�Y����G���=qCE33�.�R���ÿ@  �_
=CM�                                     Bx��  "          @Fff�&ff�Q녿�Q�����CE}q�&ff��\)�k����CO޸                                    Bx�  T          @B�\�{���������33CK33�{�Ǯ�B�\�iCTL�                                    Bx��`  "          @AG��{��ff�����\)CK��{����:�H�_�
CS޸                                    Bx���  �          @@  �!G���
=��=q��=qCM
=�!G������G����CSaH                                    Bx��Ӭ  T          @C�
�,(����ÿp����(�CI���,(����׾�p�����CO
=                                    Bx���R  �          @>{�Q쿣�
��(�����CPE�Q��Q���!�CWk�                                    Bx����  "          @<(��(���������33CSff�(����ÿ#�
�K
=C[�                                    Bx����  �          @;���
��
=��(���\)CV�q��
��Q�+��V�\C_=q                                    Bx��D  "          @Fff�\)�У׿�\)��
=CW�3�\)�z�   ��C^��                                    Bx���  �          @E��  �����ff����CZxR�  �z���!G�C^��                                    Bx��+�  �          @E��\��z�@  �a��C[���\�33>B�\@g
=C]��                                    Bx��:6  �          @G��z��z�J=q�k33C[h��z���
>#�
@:=qC]�\                                    Bx��H�  T          @I���  �z�:�H�V�HC^�=�  �
�H>�\)@��C_��                                    Bx��W�  
�          @HQ��
=��R��R�6�\Cb�f�
=�G�>�AffCc#�                                    Bx��f(  
�          @K��\)�ff�
=�.=qC_.�\)���>�
=@��C_�R                                    Bx��t�  �          @S33�5��  �5�G33CO��5��z�=L��?^�RCR\)                                    Bx��t  �          @S�
�/\)��33�Tz��g�CS  �/\)��<#�
>aG�CU�f                                    Bx��  T          @`  �0�������CY!H�0���ff>�@�=qCYL�                                    Bx���  T          @g��E����
�(���'�
CR
=�E����>aG�@c33CSz�                                    Bx��f  T          @l���S�
��p��0���-p�CL0��S�
���=u?k�CN\)                                    Bx��  "          @i���:=q����R�p�CX��:=q�
�H>��@�ffCX��                                    Bx��̲  �          @[��
=����  ���RCa�{�
=��?c�
ArffC_�3                                    Bx���X  �          @`  ������   ���C`�
������?(��A/�
C`^�                                    Bx����  T          @]p��#�
��\������HC]���#�
�{?.{A5C\��                                    Bx����  �          @Z�H�!G��\)��\�z�C]�3�!G��{?z�A(�C]xR                                    Bx��J  �          @[��%���׿z��%�CX0��%���Q�>�33@ÅCY�                                    Bx���  T          @b�\�U��Q녿aG��ep�CA�=�U����׾�����G�CF�R                                    Bx��$�  �          @h���W
=����aG��_�CE�R�W
=��\)��\)����CJ8R                                    Bx��3<  "          @i���S33���H�u�tz�CH&f�S33�\��������CL��                                    Bx��A�  "          @mp��U��(��}p��x  CH��U������R��(�CL�                                    Bx��P�  T          @p  �N{��  ��33��G�CL���N{��{������G�CQ�3                                    Bx��_.  �          @vff�N�R��(����H��{CO��N�R����\)���CT��                                    Bx��m�  �          @w��J=q�����\���CQ�J=q�(���z�����CV��                                    Bx��|z  �          @w��AG���p��������CUB��AG�������R���HCZn                                    Bx���   �          @y���>{�
�H��  ��Q�CX
�>{�   ���Ϳ�ffC\\                                    Bx���  
Z          @vff�Dz��(���p�����CT���Dz���
�8Q��,��CX�q                                    Bx��l  �          @w��AG�� �׿�{��
=CU�H�AG��=q��\)���HCZ�H                                    Bx��  T          @u�;��녿�
=����CVǮ�;��p����
����C\�                                    Bx��Ÿ  
�          @s�
�:�H�����
���CWz��:�H�(��.{� ��C[�H                                    Bx���^  "          @s33�8���  ���\�yCY�H�8���p�>\)@	��C\xR                                    Bx���  �          @u�4z��
=�����Q�C[��4z��&ff>\)@�C^��                                    Bx���  
(          @u�6ff���xQ��j=qC[��6ff�#33>�  @p��C]�\                                    Bx�� P  
�          @z=q�;��ff�����33CZ�;��%>\)@G�C]s3                                    Bx���  
Z          @xQ��,���%����
�v=qC_�q�,���0��>���@�{Ca�                                    Bx���  T          @w��#33�/\)�s33�dz�Cc#��#33�7�>�G�@�33Cdn                                    Bx��,B  �          @{�� ���7
=�xQ��d��Cd��� ���>�R>��H@�Ce�R                                    Bx��:�  
�          @|(��%��3�
��  �j�RCcs3�%��<��>�G�@��
Cd�{                                    Bx��I�  
(          @}p��!G��6ff��ff�w�Cd�=�!G��AG�>��@�
=Cf
                                    Bx��X4  
�          @y���{�4zῊ=q��(�CdǮ�{�?\)>\@���Cfz�                                    Bx��f�  
Z          @z=q�(��6ff������
=Cen�(��A�>�p�@�p�Cg0�                                    Bx��u�  �          @xQ��{�1녿�{���\Cdc��{�>{>�{@��RCfJ=                                    Bx��&  "          @s�
�Q��333�}p��q�Ce�{�Q��<(�>�ff@�Q�Cf�                                    Bx���  "          @p  �{�$z῔z�����Cb#��{�3�
>B�\@5Cd�                                     Bx��r  �          @n{����#33��{���Cb!H����1G�>k�@c�
Cd�                                     Bx��  �          @k��(��4z�aG��]�Ch33�(��9��?\)A��Ch�R                                    Bx��  I          @mp��z��.�R�u�o�
Ce���z��7
=>�ff@�
=Cf�R                                    Bx���d  	k          @l����
�/\)�fff�b=qCeٚ��
�5?�\@�33Cf�)                                    Bx���
  	�          @p  �p��,(��aG��Xz�Cc��p��1�?   @�\)Cd�                                     Bx���  T          @y���:�H��ÿ}p��j�\C[aH�:�H�$z�>�=q@}p�C]s3                                    Bx���V  "          @z�H�>�R�����\�r�\CZ
�>�R�"�\>W
=@C33C\xR                                    Bx���  
�          @x���<(�����  �n�\CZh��<(��!�>k�@Z�HC\��                                    Bx���  
�          @xQ��K��   �n{�^�HCT+��K��p�>�?��RCV��                                    Bx��%H  	�          @{��^{��Q�=p��,��CM��^{��=q>#�
@�
CO�
                                    Bx��3�  
Z          @x���C�
�(��s33�a��CW���C�
��>W
=@I��CY�                                     Bx��B�  T          @z=q�B�\�녿n{�\��CX��B�\���>�=q@\)CZ�f                                    Bx��Q:  	�          @xQ��B�\��R�k��[�CXE�B�\���>��@u�CZG�                                    Bx��_�  
�          @w
=�A��
�H�p���c�CW�H�A��
=>W
=@I��CY�f                                    Bx��n�  T          @u��<���  �s33�fffCYL��<����>u@g�C[u�                                    Bx��},  T          @u�9�����u�g
=CZ�=�9���   >�\)@���C\�\                                    Bx���  "          @vff�(���+��Q��ECaff�(���/\)?\)A�RCb)                                    Bx��x  �          @w
=�-p��&ff�h���YC_� �-p��-p�>�ff@ָRCa�                                    Bx��  T          @xQ��%�0  �O\)�B=qCb� �%�333?(�ACcG�                                    Bx���  T          @~�R�'
=�:=q�=p��+
=Cd#��'
=�9��?@  A/33Cd
                                    Bx���j  
�          @|���-p��,�Ϳ�  �j�RC`���-p��5>�
=@��Cbk�                                    Bx���  
e          @}p��0���*=q��G��l��C_�3�0���4z�>���@��Ca�)                                    Bx���  
�          @\)�/\)�.�R�u�^�HC`���/\)�6ff>�@��Cb.                                    Bx���\  
�          @~�R�5��"�\��
=��=qC]��5��333>B�\@1�C`��                                    Bx��  
�          @|(��.{�,(��p���]�C`���.{�3�
>�@�\)Ca�3                                    Bx���  
�          @�  �,���0  ��ff�r�HCa���,���:=q>�
=@�  Cc+�                                    Bx��N  
�          @\)�333�*=q��  �hQ�C_�=�333�3�
>��@�
=Ca!H                                    Bx��,�  �          @{��8���{�s33�`Q�C\���8���'�>�p�@���C^5�                                    Bx��;�  
�          @z=q�/\)�(Q�p���^�RC_�=�/\)�0��>�ff@���Ca#�                                    Bx��J@  T          @����6ff�)���}p��e�C^��6ff�333>�
=@�Q�C`�                                     Bx��X�  
Z          @��\�@  �{��(�����C[xR�@  �0��>�?�33C^��                                    Bx��g�  
�          @�=q�>�R�!G���33���C\@ �>�R�1G�>aG�@G�C^ٚ                                    Bx��v2  "          @�33�AG��\)��(���{C[xR�AG��0��>��@
=C^u�                                    Bx���  �          @���B�\�\)���H��z�C[L��B�\�0��>#�
@  C^:�                                    Bx��~  "          @�33�@  �'
=��G��e�C]
=�@  �1G�>Ǯ@�{C^                                    Bx��$  �          @�33�8���0�׿n{�QC_�3�8���7
=?
=q@��C`�                                     Bx���  
(          @�p��6ff�8�ÿs33�R=qCaaH�6ff�?\)?
=AG�CbL�                                    Bx��p  
�          @�ff�,(��B�\��=q�n�RCdz��,(��K�?
=q@�{Ce                                    Bx���  
(          @���%�@  ��=q�t��Ce+��%�H��?�@�\Cf��                                    Bx��ܼ  
�          @�p��*�H�A녿�G��_�
Cd�)�*�H�HQ�?(�A�\Ce��                                    Bx���b  	�          @�Q��%��<(��c�
�N=qCd���%��@  ?(��A�CeB�                                    Bx���  
�          @}p��p��8�ÿ�����=qCe�H�p��Dz�>�G�@˅CgT{                                    Bx���  
�          @�33�*=q�P  ��=q�f{Cf�R�*=q�W
=?(��A  Cg��                                    Bx��T  �          @�G��*�H�`�׿}p��K33Ch�\�*�H�c33?\(�A0  Ci{                                    Bx��%�  
Z          @��H�{�Tzΰ�
��\)CiT{�{�aG�?�@��Cj�                                    Bx��4�  T          @�
=�*=q�U���
��  Cgn�*=q�b�\?�@���Ci
=                                    Bx��CF  "          @�\)�z��c�
���H��z�Co���z��|(�>�=q@\��CrW
                                    Bx��Q�  �          @�33���q녿�����G�Cp�����?   @ǮCr}q                                    Bx��`�  �          @��ÿ�
=�\(�����p�Cp�f��
=�s�
>�  @S33Cs�                                    Bx��o8  "          @��׿�Q��Mp���Q���(�Cn�׿�Q��`  >���@���Cq�                                    Bx��}�  �          @�����33�Z=q�ٙ���=qCp녿�33�s�
>W
=@4z�Cs��                                    Bx��  T          @�������^�R���R����Ckff����p��>�(�@�Q�Cmp�                                    Bx���*  T          @����{�N{��33����Ch� �{�_\)>�p�@�Q�Cj�R                                    Bx����  "          @�Q�����O\)�������\Ci������a�>�{@��Ck޸                                    Bx���v  
�          @�ff�Q��O\)���H��Q�Ci�q�Q��Z=q?��@��HCk(�                                    Bx���  
�          @�p�����N{���
���Clz�����c�
>�=q@l��Co�                                    Bx����  �          @�ff���N{��z����Cj�R���_\)>�p�@�(�Cl��                                    Bx���h  �          @��R�  �G
=��Q����
Cj(��  �b�\=��
?��Cm�{                                    Bx���  �          @�  ��Q��u���=q��G�Cs+���Q��\)?B�\A
=Ct\                                    Bx���  "          @��� ����
=�k��2{Ct��� �����
?�ffA~{Cs��                                    Bx��Z  
�          @���Q��hQ�z�H�R�\Co�
�Q��hQ�?s33ALQ�Co��                                    Bx��   "          @�����{����H���Cn�3����Q�>�@�z�Cp�                                    Bx��-�  
�          @�
=����x�ÿ�����=qCo���������?L��A��Cp��                                    Bx��<L  T          @�{��{�w
=��ff�\Q�Ct8R��{�w
=?��\AU�CtE                                    Bx��J�  	�          @�����
=�n{�p���J�\Cr���
=�l(�?��A`��Crff                                    Bx��Y�  	�          @�녿�p��mp��k��E��Cq����p��j�H?��Ac�Cq��                                    Bx��h>  T          @�G��	���xQ쿂�\�O33Cq
=�	���w�?��AW\)Cp�R                                    Bx��v�  
Z          @��R�z��q녿����j=qCqQ��z��u?k�A>ffCq��                                    Bx����  
�          @�p�����tz῕�v=qCs�=����x��?fffA<(�Ct8R                                    Bx���0  
�          @���(���n{�\(��+
=Cj��(���i��?�\)A_�Cj(�                                    Bx����  �          @���{�c�
�Q��,(�Ck.�{�_\)?���Aa�Cj��                                    Bx���|  T          @����\�R�\�aG��Ep�Ck8R��\�R�\?c�
AIp�Ck.                                    Bx���"  
(          @����aG��fff�AG�Clc���_\)?}p�AS
=Cl:�                                    Bx����  
�          @�����\��녿z�H�>�\Cp����\��Q�?�Q�Ag�Cp5�                                    Bx���n  T          @�����n�R���\�NffCl�����n�R?}p�AIp�Cl��                                    Bx���  "          @���333�fff�G���RCh��333�`  ?��AdQ�CgY�                                    Bx����  
�          @���AG��XQ�B�\���Cd8R�AG��S�
?��AQ�Cc�{                                    Bx��	`  T          @�Q��;��i���W
=�#�
CgJ=�;��e�?�{AX(�Cf��                                    Bx��  
�          @�\)�8���e�����Pz�Cg{�8���hQ�?c�
A-�Cgs3                                    Bx��&�  T          @�����\�mp���  ��  CnQ���\�u�?J=qA!�Co&f                                    Bx��5R  �          @�ff���r�\��{�g\)CqǮ���u�?uAF�HCr
=                                    Bx��C�  	�          @����
=q�|(���Q��m�CqG��
=q��  ?xQ�A?�
Cq��                                    Bx��R�  
�          @�33�z��w���\)��z�Cq�H�z�����?G�A�
Cr�
                                    Bx��aD  
�          @��\��p��p�׿��\���
CuT{��p��x��?O\)A-�Cv                                    Bx��o�  
�          @�=q��
=�p�׿����33Cu���
=�y��?J=qA(��Cv��                                    Bx��~�  �          @��
����o\)��p���p�CsE����vff?Tz�A/\)Cs�                                    Bx���6  �          @��Ϳ����`�׿�Q���\)Cr������g�?@  A'�Cr�f                                    Bx����  
Z          @�  ����l�ͿaG��?�Cr�����hQ�?�\)At  Cr��                                    Bx����  	�          @�(����`  ��G��c�Cq5ÿ��a�?fffAJ�RCqk�                                    Bx���(  �          @����Q��[��s33�T  Cn��Q��[�?k�AMG�Cn(�                                    Bx����  
�          @�����R�X�ÿTz��8Q�Cl����R�U?�G�AaCl@                                     Bx���t  
Z          @������c33�L���*=qCl����]p�?�{Al��Ckc�                                    Bx���  �          @�����\�y������^�RCrn��\�z=q?��AUG�Cr�                                     Bx����  
�          @�\)�#�
���\��(�����C��\�#�
��z�?��@��C��{                                    Bx��f  �          @�{>�ff���ÿ��
��  C�/\>�ff��(�?�@�{C��                                    Bx��  J          @��
>\)������\)���
C��>\)���?+�A{C��{                                    Bx���  
�          @��\��{��\)��
=����C��ᾮ{��  ?5A\)C��q                                    Bx��.X  "          @��Ϳ���������HC��H����
=?�G�AIp�C��
                                    Bx��<�  
�          @�  >��
���ÿ�����C�G�>��
��{>��@�\)C��                                    Bx��K�  ^          @��
�����z�u�-G�Cp�������Q�?�z�A�=qCp
                                    Bx��ZJ  "          @��
�z���{��ff�=�Crk��z���33?�{Aw33Cq��                                    Bx��h�  
�          @���\)��G��p���(z�Cs��\)��z�?�G�A���Cr��                                    Bx��w�  	�          @�33����
=�^�R�{Cs����G�?��A�ffCr�                                    Bx���<  
�          @����Q����\�Y���Q�CqE�Q����?��RA��CpE                                    Bx����  
�          @��H�!������&ff��(�Co�{�!�����?�33A��HCm�f                                    Bx����  	�          @��\�<(���(��B�\�(�Cj�=�<(��i��?���A��RCg)                                    Bx���.  "          @����������#�
��ffCwn������?��
A�Cu�3                                    Bx����  |          @��7��g
=>�p�@��HCg���7��:=q@
�HA߮Can                                    Bx���z  
�          @��H�"�\�s33���
�k�Cl:��"�\�QG�?�
=A�(�Ch(�                                    Bx���   
�          @������{���  �O\)Cp�����`  ?�ffA�=qCn
=                                    Bx����  
�          @��H��
��녾�����Cs���
�n{?�Q�A�\)Cq�                                    Bx���l  
�          @��
�u��   >\@���CU&f�u�����?�{A��\CN�q                                    Bx��
  "          @����������>��@���CP}q������
=?�p�A��HCJ�                                     Bx���  
�          @���o\)�(��?8Q�A�CW5��o\)���?�(�A�  CN�{                                    Bx��'^  �          @���Z=q�N�R?8Q�A
=C_s3�Z=q���@�A��CW&f                                    Bx��6  
�          @�(��w
=�\)��=q�W
=CT���w
=�G�?�ffAR�HCRz�                                    Bx��D�  
�          @���K��AG�>�33@���C_}q�K��=q?���A�
=CY!H                                    Bx��SP  
�          @��
�K��C33?z�HAG�
C_�=�K��
=@=qB �CU�{                                    Bx��a�  "          @���	���O\)>���@�=qClu��	���'
=?���A�  Cf��                                    Bx��p�  
          @��׿��\�]p������Cy�)���\�p��>�@�(�C{Y�                                    Bx��B  ^          @��R��=q�k������t��Cv���=q�mp�?uAUp�Cv�f                                    Bx����  �          @������n�R�xQ��[�
Cy������l��?��Av�HCyY�                                    Bx����  �          @�=q��
=�p  �k��ECrǮ��
=�l(�?��Au�Crc�                                    Bx���4  
�          @�33��
�qG�����\)CqO\��
�`  ?�  A��Coz�                                    Bx����  
�          @�Q��C33�N�R?�@�(�Cb���C33�   @
=A�33C[O\                                    Bx��Ȁ  �          @����Q��2�\?��A�C\\)�Q녿�(�@#33B
�CO�f                                    Bx���&  "          @��
�
=q�n{>�@�p�Co��
=q�<(�@B�
Ci��                                    Bx����  �          @��
���o\)=�Q�?�  Co�q���I��@G�A��
CkO\                                    Bx���r  �          @�Q��N�R�>{?�  AM�C^���N�R��@��B �CT!H                                    Bx��  
�          @�����p�׼���(�CnE���O\)?�AͮCjJ=                                    Bx���  
�          @����
=��33��\)�aG�Cr���
=�b�\@�
Aԣ�CoB�                                    Bx�� d  �          @�33���R���=�G�?�ffCu�f���R�mp�@Q�A�\)Cq�                                     Bx��/
  	�          @��Ϳ���Q�L���33C�q����p�@33A߅C~c�                                    Bx��=�  
(          @��������H�=p���C}�������?޸RA�
=C|�                                    Bx��LV  
�          @�{��
=����u�=G�C쿗
=����?�(�A���C~�
                                    Bx��Z�  	�          @�{������������ffC}𤿙����
=?��AX��C~+�                                    Bx��i�  "          @�
=�:�H�~{�=q��=qC���:�H����<#�
=uC���                                    Bx��xH  
(          @��׾�Q�������뙚C��H��Q���
=>\)?�p�C��{                                    Bx����  T          @��׾�p���{��\����C���p���?k�A)G�C��q                                    Bx����  
Z          @�  ��=q���Ϳ�ff���C�XR��=q���?^�RA ��C�o\                                    Bx���:  �          @�  ��=q��=q����z�C�` ��=q����?s33A/�C��                                    Bx����  
�          @�=q�������p���C~�H�����Q�?�AT��C~�f                                    Bx����  T          @�=q�������R����lQ�C~Ϳ�����ff?���Av�\C~                                      Bx���,  
�          @��������׿�{��ffC녿�����{?z�HA6�HC�7
                                    Bx����  "          @��ÿ�ff���Ϳ�
=��33Cy�{��ff��G�?p��A:{Cz.                                    Bx���x  �          @����G��w
=�
=q���
C{�ῡG���p�>.{@C~#�                                    Bx���  T          @�33��Q���=q�޸R���\C}����Q���(�?!G�@�C~�=                                    Bx��
�  T          @��R��ff��  ��G��x��Cy���ff����?��A`��Cz!H                                    Bx��j  "          @��ÿ�����ff��{�V�RC}��������
?�\)A�G�C|ٚ                                    Bx��(  "          @�Q�aG���=q�E���\C��ͿaG�����?�(�A�\)C�8R                                    Bx��6�  T          @��>����(�=L��?!G�C���>���{�@p�B {C�)                                    Bx��E\  "          @��H>\����?��@�ffC�aH>\�dz�@5�B  C��                                    Bx��T  	�          @����Q����H    =uC��f��Q��l��@G�A�
=C�@                                     Bx��b�  
�          @�33�^�R���R���Ϳ��
C�|)�^�R�w
=@\)A�=qC��f                                    Bx��qN  
Z          @�녿�(����ÿ����ffC|  ��(���?���A�\)Cz�
                                    Bx���  
�          @�p����
��33�J=q�(�C{�H���
���H?�p�A��RCz�
                                    Bx����  	�          @�33������33��R����C�w
������  ?�\)A���C�f                                    Bx���@  	�          @�������u�H��C������y��@ffA��HC�7
                                    Bx����  
�          @���^�R��ff�����C�~��^�R���\?���A��C��                                    Bx����  
Z          @��;���{>L��@.�RC�þ��^{@
=B�
C��                                    Bx���2  �          @��
���
��zᾙ���~{C|�{���
�l(�?�z�A�ffCz�)                                    Bx����  �          @�\)���R���Ϳ��
�V�RC}aH���R���\?��A�z�C}�                                    Bx���~  
�          @�
=��p���  �����C}�쿝p��z�H?�  A�
=C|�                                     Bx���$  T          @���c�
��Q쿆ff�X(�C���c�
��?��A�
=C�H                                    Bx���  �          @�
=������
�=p��=qC{O\����y��?\A��
CzQ�                                    Bx��p  T          @�p��(��q녿���  Co�H�(��_\)?��A��Cm޸                                    Bx��!  �          @���ff�~{����\(�Cx���ff�{�?�Q�A{\)Cxz�                                    Bx��/�  
Z          @�33��G��x�ÿ\(��A�Cff��G��qG�?�ffA�p�C~�3                                    Bx��>b  
�          @|�Ϳ\(��e��������C�E�\(��q�?0��A!G�C��)                                    Bx��M  
Z          @��������z�H��\)��
=C��=������{?+�AC��                                    Bx��[�  T          @�=q�
=��=q��{����C����
=��?xQ�AN�HC��                                    Bx��jT  "          @����c�
�|�Ϳ�Q�����C��Ϳc�
���
?W
=A4z�C���                                    Bx��x�  |          @��R��p��~�R��\)��{C|ٚ��p���\)?333A�C}�
                                    Bx����  
�          @��׿W
=���ÿ��
�ap�C�ÿW
=�~{?�p�A��HC�H                                    Bx���F  
�          @�Q쿏\)�z=q���\��ffC}�῏\)�\)?xQ�AR�\C~L�                                    Bx����  "          @�\)�����(���\)��
=C�쿅����?}p�AMp�C��                                    Bx����  
�          @�{�.{���H��ff��Q�C�LͿ.{��G�?Q�A+
=C��f                                    Bx���8  
�          @�(��xQ���\)������Q�C���xQ���{?Y��A)��C���                                    Bx����  "          @�
=�5�x���33����C���5��(�>���@tz�C�l�                                    Bx��߄  
�          @��׿!G��z=q�
�H��{C�s3�!G����R>L��@"�\C��                                    Bx���*  "          @��\����u��{���HC~ٚ������?#�
A��C��                                    Bx����  
�          @�p������h�ÿ�G�����Cv5ÿ����p��?Tz�A8��Cv��                                    Bx��v  "          @������_\)�У���Q�CuQ�����u�>�G�@��CwE                                    Bx��  �          @\)��  �L(���
=����CqT{��  �^{>�@�\)CsG�                                    Bx��(�  J          @|���333�(�ÿ@  �1�C_O\�333�(��?=p�A/
=C_W
                                    Bx��7h  
�          @��\�`�׿�
=�.{��\CP�\�`���   >�(�@ÅCQ��                                    Bx��F  
2          @��H�g
=��녾u�UCO��g
=�޸R?J=qA2�RCM��                                    Bx��T�  
�          @����o\)���>���@�Q�CHff�o\)��G�?�G�AjffCC)                                    Bx��cZ  �          @��\�C�
�0�׿��
��(�C^
�C�
�AG�>�Q�@���C`��                                    Bx��r   
�          @�G��O\)�!녿�
=��CY���O\)�1G�>���@��
C\�                                    Bx����  �          @�z��p�׿���>aG�@HQ�CK  �p�׿��R?��
Ah��CFE                                    Bx���L  
Z          @���z=q���>���@���CF��z=q�h��?��
AeCA
=                                    Bx����  
Z          @��|(����
��p����
CF
=�|(����>�Q�@�{CF
                                    Bx����  �          @�Q��z=q����s33�O�CF���z=q��{����\)CJs3                                    Bx���>  �          @����q녿����H���CH���q녿�p��   ����CO�)                                    Bx����  �          @�
=�~�R��녿B�\�'33CC�R�~�R��\)�L�Ϳ333CG                                      Bx��؊  
�          @�ff���
�B�\<�>�p�C>s3���
�!G�>�
=@�G�C<�R                                    Bx���0  "          @�Q�����E����ƸRC>xR����fff<#�
>#�
C@(�                                    Bx����  T          @�(���=q���
=q����C:!H��=q�.{�W
=�1�C<��                                    Bx��|  T          @����  �����p���p�C<��  �333        C=ff                                    Bx��"  �          @�  ��(��8Q�
=��C=���(��n{���ٙ�C@��                                    Bx��!�  �          @���G��5�L���0��C=�3��G����\���R��\)CB8R                                    Bx��0n  T          @�(��{����
�E��+�CB���{����
��G��˅CF
=                                    Bx��?  �          @��R���׿�  �333��CA�3���׿�(���\)�}p�CD޸                                    Bx��M�  "          @����u����p���S�
CD�R�u����R�B�\�'�CI@                                     Bx��\`  T          @�
=��Q����H���rffCz�H��Q���33?�33Amp�Cz��                                    Bx��k  
�          @�G����
��=q�����  Cyff���
����?��\AP��Cy�                                    Bx��y�  �          @�녿\��33�����  Cy���\��p�?�ffAUp�Cy��                                    Bx���R  �          @�\)�����Q쿡G�����Cx�ÿ����=q?��
AUCyB�                                    Bx����  "          @�  �˅�k���Q���ffCv���˅�x��?5Az�Cw�=                                    Bx����  �          @��Ϳ�(��g
=��{���C{Y���(��z�H?�@�G�C|��                                    Bx���D  �          @��
�Q��l(�������C����Q���  ?
=q@��HC�7
                                    Bx����  "          @��\���j�H��G���\)C�׾���G�>�(�@�Q�C��                                    Bx��ѐ  "          @�ff�L���x�ÿ�����33C����L����(�?5A(�C��                                    Bx���6  �          @�p���\)�|�Ϳ�  ��=qC��H��\)����?�G�Ab�\C���                                    Bx����  �          @�p���
=�vff����mG�C}���
=�u�?�{Ax��C|�3                                    Bx����  ^          @�����
�\)�����C|B����
�j�H?�33A��
Cz��                                    Bx��(  
Z          @�{��ff�n�R�
=� ��Ct33��ff�^�R?��HA�\)Cr��                                    Bx���  
Z          @�����l(�����=qCr�3����X��?��
A�z�Cp�                                    Bx��)t  �          @�G���(��p  ����ffCr=q��(��^{?�G�A�{CpaH                                    Bx��8  �          @���
=�g���G��Yp�Co���
=�g
=?��A_
=Co�f                                    Bx��F�  
�          @�\)��\)�j=q�}p��XQ�Cr��\)�h��?�=qAm�Cr�                                    Bx��Uf  T          @��R�   �^{��=q��\)Cp!H�   �j=q?0��A=qCqaH                                    Bx��d  �          @�=q�����c�
�˅��p�Cr�3�����w
=?�@��Ct��                                    Bx��r�  T          @�(���(��p�׿�p���  Cun��(��~�R?5A��Cv��                                    Bx���X  T          @�p��  �@�׿�33����Ci5��  �Z�H>8Q�@p�Cl��                                    Bx����  �          @�ff��(��U����R���
Con��(��_\)?333ACp��                                    Bx����  	�          @���Dz��C33>�
=@��C`��Dz���?��HA�(�CY�                                    Bx���J  
�          @�Q��*=q�g
=>L��@#33Ci���*=q�=p�@�
A�
=Cd�                                    Bx����  �          @���\(��<(�>��@�C\���\(���\?��A��HCU�)                                    Bx��ʖ  
�          @��\�Tz��E>�ff@�z�C^�R�Tz����@   AУ�CW�{                                    Bx���<  �          @�=q�b�\�2�\?+�A33CZQ��b�\��@�\A�ffCQ�{                                    Bx����  T          @�G����ÿ�?0��A��CL�����ÿ��H?�ffA��
CD�                                    Bx����  
�          @�
=�mp��ٙ��W
=�8Q�CL�\�mp���ff?5A!�CJ��                                    Bx��.  "          @����QG��'�?Tz�A3
=CZ�3�QG���ff@�A��CPǮ                                    Bx���  �          @�=q��4z�>��H@�ffCfJ=����?�33A�\)C^n                                    Bx��"z  T          @��
�:=q�޸R�0����\CR�{�:=q�:�H��Q���Ca&f                                    Bx��1   �          @���7��p�����CY��7��8�þ��׮Ca&f                                    Bx��?�  T          @�z��>{�.{��p���(�C^k��>{�N{�L�Ϳ+�CcL�                                    Bx��Nl  
�          @��H�,���/\)�������Cau��,���S33�\)��z�Cf�=                                    Bx��]  �          @��H�.�R�p��p��CZ�f�.�R�J�H�n{�I��Ce.                                    Bx��k�  T          @��\�1��ff��(����
C\@ �1��1G����
��\)C`�3                                    Bx��z^  �          @���333�N�R>L��@,(�Ce\�333�(��?�{A�ffC_\)                                    Bx���  
�          @���L���6ff��Q����C]�3�L���'
=?�Q�A��\C[@                                     Bx����  �          @�=q�Mp��8Q�k��C�
C]��Mp��$z�?��A���CZ�                                     Bx���P  "          @�G��S33�*=q���
���\CZ�H�S33��?�\)AtQ�CXn                                    Bx����  �          @���8Q��E�>�\)@vffCb�H�8Q���R?���A�{C\��                                    Bx��Ü  T          @�(��:�H�Mp�>�
=@�Q�Cc���:�H� ��@�\A���C\                                    Bx���B  �          @���� ���h��?�G�AMCkk�� ���$z�@0��BG�Ca��                                    Bx����  "          @�=q�9���`��>�  @L(�Cfn�9���6ff@33A�  C`�{                                    Bx���  �          @��H�AG��\��>�\)@\(�Cd�q�AG��2�\@�\Aң�C^��                                    Bx���4  �          @���z�H���>�(�@�\)CR  �z�H�ٙ�?�ffA��
CKxR                                    Bx� �  
�          @��
�s�
� ��?&ffA=qCUaH�s�
��ff?�\)A�\)CMG�                                    Bx� �  
2          @�G��X���>{>�{@�Q�C]L��X���
=?�A�Q�CV�H                                    Bx� *&  "          @���5�Z�H��Q쿔z�CfO\�5�=p�?��HA�
=Cb:�                                    Bx� 8�  T          @�p���z��`  ����ffCqff��z���ff�#�
����Cu��                                    Bx� Gr  "          @��R�Q��e���܏\Co\)�Q���ff<��
>aG�Cs#�                                    Bx� V  
�          @����Q��k���p���p�Cr.��Q���p�>k�@:=qCu
=                                    Bx� d�  
�          @�=q��\�g
=�����Cs�f��\��ff=u?O\)Cw#�                                    Bx� sd  "          @�z�Ǯ�mp���\��\)Cw8R�Ǯ���
���
�k�CzW
                                    Bx� �
  "          @��׿�ff�^�R�  ���HCr�)��ff��z���Ϳ��Cvz�                                    Bx� ��  T          @�Q쿫��R�\�ff�
=Cwٚ������þ�z��z=qC{�H                                    Bx� �V  "          @�G���ff�J=q�%��G�Cw�f��ff��G������C|.                                    Bx� ��  "          @�녿�ff�Z=q�&ff�  Cu� ��ff���׾�G���G�Cy��                                    Bx� ��  
(          @��H�fff�,(�?.{A	CX��fff��Q�?�p�A��HCPW
                                    Bx� �H  "          @�p��[��/\)>���@�{CZ���[��Q�?�\A��CS�
                                    Bx� ��  
�          @�{�^{�0��=�G�?���CZ� �^{�33?��
A�\)CU�=                                    Bx� �  T          @��\�XQ��,�ͽ���\)CZ���XQ���?�ffA��CV��                                    Bx� �:  
(          @����U��-p��\)��\)C[&f�U����?��A�  CW�f                                    Bx��  �          @����h���  �W
=�333CS��h����?}p�AU�CQ)                                    Bx��  �          @���`����R>#�
@ffCW8R�`����?�A��RCR
                                    Bx�#,  �          @�G��R�\�.{��
=���
C[��R�\�"�\?�ffAc�CY��                                    Bx�1�  �          @�{�N{�&ff�(���CZ��N{�"�\?Tz�A7�CZB�                                    Bx�@x  �          @�ff�C�
�(�ÿ����z�\C\�{�C�
�6ff>��@���C^�3                                    Bx�O  �          @����Dz��*=q��33���C\��Dz��@  >B�\@"�\C`Y�                                    Bx�]�  �          @��R�AG��0  ��G��^{C^\)�AG��8Q�?�@��C_��                                    Bx�lj  �          @�z��?\)�#�
��Q����RC\�{�?\)�333>��R@�33C_0�                                    Bx�{  �          @���I���,��>���@�=qC\�)�I�����?�Q�A��CV&f                                    Bx���  �          @���B�\�:�H?B�\A"�RC_���B�\�
=@
=qA�RCV�R                                    Bx��\  �          @��R�7��U���\���CeB��7��G
=?��A��CcW
                                    Bx��  �          @���<���N�R�Q��(��Cc�
�<���L(�?z�HAK33Cc0�                                    Bx���  T          @�p��I���B�\��ff���C`��I���5�?���A|z�C]��                                    Bx��N  �          @�\)�U��;�����G�C]L��U��333?��
AUp�C\                                    Bx���  �          @��H�B�\�<�Ϳ\(��6�HC`
�B�\�=p�?J=qA)G�C`B�                                    Bx��  �          @���H���;�����p�C^�q�H���1�?���AeG�C]u�                                    Bx��@  �          @��R�J=q�.{>�{@�33C\�3�J=q�	��?ٙ�A�33CVB�                                    Bx���  �          @����N�R�1G�>u@N{C\�\�N�R�\)?��A���CV                                    Bx��  �          @��
�Vff�2�\��Q쿜(�C[�\�Vff�(�?�\)A��
CX
=                                    Bx�2  �          @����Z=q�0  �\��ffCZ��Z=q�#33?���Ag�CX�{                                    Bx�*�  �          @�z��Z=q�HQ�B�\�=qC^�=�Z=q�1�?���A�  C[.                                    Bx�9~  �          @�p��\(��G
=����
=C^{�\(��;�?�z�AeG�C\n                                    Bx�H$  �          @�\)�Vff�;�����
=C]:��Vff�0��?���Ac�C[��                                    Bx�V�  �          @�Q��fff�B�\�����z�C\.�fff�8��?���AVffCZ��                                    Bx�ep  �          @��p  ��ff���
�^�RCM���p  ��=L��?!G�CP�q                                    Bx�t  �          @�
=�u��������C;��u���R��\)�y�CEٚ                                    Bx���  �          @�ff�vff��Ϳ�=q����C<��vff���
�����lz�CFp�                                    Bx��b  �          @�\)��Q�p�׿�Q����HCA=q��Q��
=�xQ��Hz�CJ�                                     Bx��  �          @��w
=>8Q�Ǯ��=qC1Q��w
=�&ff��
=���HC=�\                                    Bx���  �          @��w
=>�\)��{���\C/ٚ�w
=�
=�\���\C<��                                    Bx��T  �          @��
���R�#�
��\)��  C4}q���R�}p��˅��G�CAL�                                    Bx���  �          @�{��(�>��
��33��ffC/����(��녿�=q��33C;n                                    Bx�ڠ  �          @�����z�aG��������HC6ٚ��zῑ녿�p���Q�CB�\                                    Bx��F  �          @�G��������
=����C:�����������d��CD��                                    Bx���  �          @��\���@  �޸R���C=�H�������{�T��CG(�                                    Bx��  �          @��H��=q�B�\�Ǯ���HC6W
��=q�u��  �n�HC?��                                    Bx�8  �          @�����\��=q����p�C7Y����\�u��=q�P  C?�\                                    Bx�#�  �          @����p���\)��  �o�
C7u���p��aG��n{�1G�C>��                                    Bx�2�  �          @�G��������Ϳ����G�C9\��������}p��>�RCA
                                    Bx�A*  �          @��\��33>�ff��\)��  C.^���33���R��z����
C7�H                                    Bx�O�  �          @�33��p�?�Ϳ�z��^{C-W
��p������
�t��C5�\                                    Bx�^v  �          @������\?zῧ��}��C,���\�.{����{C6#�                                    Bx�m  �          @�  ���R?W
=��\)��Q�C)O\���R<��
��{���\C3�3                                    Bx�{�  �          @�=q���?^�R����xz�C):����=�Q��ff��ffC2�                                    Bx��h  �          @�G�����?�녿�{�W33C%޸����>�녿�����C.�
                                    Bx��  �          @�����
?��ÿ�  �u��C#+����
?   ��  ��C-�                                    Bx���  �          @�
=��  ?�
=��(���\)C!aH��  >��   ��(�C-�\                                    Bx��Z  �          @�����
=?����ff���C"�=��
=>L���{��p�C18R                                    Bx��   �          @�G���  ?�������Q�C 
=��  ?
=q�
=��C,��                                    Bx�Ӧ  �          @�
=��\)?�{�\��G�C":���\)>Ǯ�   �ɮC.�R                                    Bx��L  �          @�(�����>���������C1�3�����.{�������
C=�                                    Bx���  �          @�p�������R���
���CD�������p��fff�1CM�f                                    Bx���  �          @��R����  ��=q����C7@ �����\��p��rffC@��                                    Bx�>  �          @�����G�=��Ϳn{�8Q�C2����G��\�Y���(��C8�{                                    Bx��  �          @�
=���\>�=q���
�J�RC0����\��zῂ�\�I�C7��                                    Bx�+�  �          @�
=��=q?k��(����C(����=q>�ff����K33C.Q�                                    Bx�:0  �          @������ͽ��
�����T��C4�����Ϳ!G��h���/
=C;��                                    Bx�H�  �          @�G���{��Q�\(��&{C8h���{�@  �\)��
=C=�                                    Bx�W|  �          @�\)���H��(��h���2�RC9T{���H�Tz����{C>B�                                    Bx�f"  �          @�p���  ���W
=�%�CIff��  ��{=�?��
CK�H                                    Bx�t�  �          @�������p���ff�NffCMh�����\)=�Q�?���CPJ=                                    Bx��n  �          @�
=��=q��
=�aG��+�CI@ ��=q���=�Q�?���CK��                                    Bx��  �          @�p���{��\)�O\)� ��CL\��{��>u@@  CM��                                    Bx���  T          @�p���\)�ٙ��u�>�RCI޸��\)����<�>�{CL�                                    Bx��`  �          @�����  ���\(��*{CIxR��  ��\)=���?��CK�\                                    Bx��  �          @��H��\)�����}p��H  CF�3��\)��  �����CJ�                                    Bx�̬  �          @�z����ÿ�=q�J=q��CHE���ÿ�\=�G�?���CJh�                                    Bx��R  �          @�����\)�\���H�ǮCG����\)�Ǯ>�{@�=qCH8R                                    Bx���  �          @�33��
=���
�u�=p�CJ޸��
=��z�?0��A
�HCIp�                                    Bx���  �          @�������������uCOT{���ÿ���?J=qA33CMٚ                                    Bx�D  �          @�  �r�\�
�H�����[�CQ�\�r�\�=q>.{@
�HCT}q                                    Bx��  �          @��R�u����H��ff�Z�\CO+��u��{=��
?�=qCR(�                                    Bx�$�  �          @���l���33��G��W�
CP�3�l����>��?��RCS�H                                    Bx�36  �          @���w����H�z�H�P��CK�H�w���p�<#�
>.{CO�                                    Bx�A�  �          @�Q���G���\�k��<(�CK����G����R=�Q�?���CNE                                    Bx�P�  �          @��������  �\(��.ffCK������Q�>�?�CMQ�                                    Bx�_(  �          @�33��
=��
=�G��\)CI����
=��>.{@z�CK��                                    Bx�m�  
�          @�Q���  ��
=�B�\�\)CM����  �33>��R@y��CO)                                    Bx�|t  �          @���p���Q�Tz��-�CQ���p�����>�33@���CR��                                    Bx��  �          @�(��n{�p��333��HCR�q�n{���?   @љ�CSW
                                    Bx���  �          @��H�j=q��R�8Q���CSO\�j=q��\>��H@�  CS�q                                    Bx��f  �          @�  �^{��R�Q��4z�CT���^{�>���@��RCV                                    Bx��  �          @�=q�U�$z�xQ��N�RCY}q�U�-p�>�G�@��
CZ�R                                    Bx�Ų  �          @���H���4z�Tz��2�RC^  �H���7
=?0��A�RC^ff                                    Bx��X  �          @���U�'��+���CZ��U�'
=?8Q�A33CY�3                                    Bx���  �          @�=q�L(��6ff�
=���
C]��L(��1G�?h��AAC\�                                    Bx��  �          @����R�\�,�Ϳ
=q����C[Y��R�\�'
=?^�RA;33CZn                                    Bx� J  �          @�=q�[��%��
=���HCY
=�[����?k�AE�CW��                                    Bx��  �          @�  �]p��p�>L��@(��CWc��]p���?��A���CRz�                                    Bx��  �          @����`����H���
���CV���`�����?���Aw
=CSh�                                    Bx�,<  �          @����0  �^{�h���9Cg���0  �^{?n{A>{Cg�\                                    Bx�:�  �          @�33�;��K������{CcaH�;��>{?�(�A�ffCac�                                    Bx�I�  �          @�Q��/\)�N{������Ce�f�/\)�A�?�Q�A��
Cc��                                    Bx�X.  �          @���"�\�U��(����RCh���"�\�N{?�=qAk
=Cg�                                    Bx�f�  �          @���5��b�\��R���\CgaH�5��XQ�?��HAu�Cf�                                    Bx�uz  �          @�=q�(���XQ��ff����Ch��(���J�H?��
A��Cf.                                    Bx��   �          @�  �.{�b�\���ÅChu��.{�S�
?�=qA�  Cf�H                                    Bx���  �          @����"�\�k��\)��CkaH�"�\�^�R?�=qA�G�Ci�
                                    Bx��l  �          @�G���H�o\)�L���#\)Cm���H�i��?��Aj{Clc�                                    Bx��  �          @�p���p��\�Ϳ0���{Cp#׿�p��U?��A{�
Co\)                                    Bx���  �          @�\)�\)�\��>.{@
=Cl�3�\)�:�H?�A���Chs3                                    Bx��^  T          @�
=�\)�>�R?�Q�A��RCf��\)� ��@   B33CZ��                                    Bx��  �          @���� ���E?�\)A��Cf��� ���G�@-p�B G�CZ�=                                    Bx��  �          @�  ���H�1�@(�B�RCjǮ���H��(�@_\)Ba��CS�                                    Bx��P  �          @�33�!��Z�H?�@�p�Ciz��!��,��@
�HA�Cb�{                                    Bx��  �          @�
=�<(��Vff=L��?z�Cd��<(��9��?�
=A�
=C`�=                                    Bx��  �          @����Fff�U��\)�޸RCc�Fff�>{?�G�A���C_                                    Bx�%B  �          @����A��X�ý��
��ffCd0��A��@  ?˅A��C`��                                    Bx�3�  �          @����C�
�W
==�Q�?���Cc���C�
�8��?�(�A�p�C_^�                                    Bx�B�  �          @���G
=�U�=L��?+�Cb�R�G
=�8Q�?�A�=qC^��                                    Bx�Q4  �          @���G��Tz�<��
>��Cb�G��8��?��A�z�C^Ǯ                                    Bx�_�  �          @�  �S�
�>�R��������C]�q�S�
�333?���Ad(�C\5�                                    Bx�n�  �          @�  �a��z������ffCRY��a��,�Ϳ(�����CYu�                                    Bx�}&  T          @���Mp���z���R��CR���Mp��7����R��
C]�=                                    Bx���  �          @���N�R��p��#�
���CP33�N�R�0�׿�33���C\�                                     Bx��r  �          @�
=�Y������ ���	Q�CJ(��Y�����\���CW��                                    Bx��  �          @�{�G
=��ff�5� p�CJ�H�G
=� �׿�������CZ��                                    Bx���  �          @�  �K��Q��Dz��,z�CBn�K������\��33CV�                                     Bx��d  �          @���Mp�����8Q��&��C;Y��Mp���G��z��p�CP��                                    Bx��
  �          @�z��U�\(��*�H���CBxR�U�33�����HCS}q                                    Bx��  �          @�Q��`�׿n{�(���
=CB���`�����{��CR��                                    Bx��V  �          @�G��c33����'
=��HCD���c33�(����
���RCS�                                    Bx� �  T          @��\�mp��=p��%��	(�C?Y��mp���33��33��(�CO{                                    Bx��  �          @���u��&ff�����ffC=�f�u���(����
��p�CL0�                                    Bx�H  
�          @�\)�w��Q�����C?�3�w���(���
=��
=CK�3                                    Bx�,�  T          @�{�}p��fff�������C@�)�}p���zῑ��m��CJ��                                    Bx�;�  T          @�  �|�Ϳ�
=��ff��(�CD���|�Ϳ�33��G��N�RCM�                                    Bx�J:  �          @�  ��  ���Ϳ�G����
CCp���  ����G��O�
CLc�                                    Bx�X�  �          @�ff���ÿ+���Q����C=� ���ÿ�33��z��r�RCG+�                                    Bx�g�  �          @�=q���\��ff�˅����CE�����\��33�E���CL�q                                    Bx�v,  �          @�����
��p���
=����CG�����
���R����ڏ\CM                                    Bx���  �          @�33��\)�s33��ff��\)C@���\)��=q�h���5��CHs3                                    Bx��x  �          @������H�xQ��
=��{CAT{���H��33��G��N�\CJ�                                    Bx��  �          @�����  ����(���33CDY���  ���Ϳs33�B�\CL�{                                    Bx���  T          @����fff���R�\)���CJ�\�fff�Q쿠  ���
CU�                                     Bx��j  �          @����n{�����	����z�CI@ �n{�33�����uCS�                                    Bx��  T          @����[���\)�����\CMJ=�[��%�����
=CY
                                    Bx�ܶ  
�          @��HQ��G��$z��{CQO\�HQ��1G�����(�C]p�                                    Bx��\  T          @��Mp���G��(�����CM=q�Mp��%���{���\CZ޸                                    Bx��  �          @�{�J=q��z��7��!�CH#��J=q�Q��(����
CX�R                                    Bx�	�  
�          @��>{���C33�.�CI���>{��R�Q���\C[�{                                    Bx�	N  �          @�33�5��  �Y���>z�CK�=�5�-p������C_��                                    Bx�	%�  �          @�p��.{���R�e��H�\CLz��.{�2�\�$z���Ca��                                    Bx�	4�  �          @�p��,�Ϳ����g
=�J�RCK��,���0���'
=�Q�Ca�H                                    Bx�	C@  �          @�
=�1녿���e��E�RCMO\�1��6ff�"�\�Q�Ca�                                     Bx�	Q�  �          @�
=�>{����Z�H�9�RCK�q�>{�0�������ffC^�f                                    Bx�	`�  �          @��H�HQ�E��P���5  CA�H�HQ��{� �����CWE                                    Bx�	o2  �          @��R�S33�=p��Q��0�C@���S33�(��#33��CU�H                                    Bx�	}�  �          @���W��p���Q��,��CC���W��������=qCW)                                    Bx�	�~  �          @����[��L���N{�)�CA+��[��{�{��G�CT�3                                    Bx�	�$  �          @�Q��`  �Tz��G��#�CAc��`  ����
=��(�CT!H                                    Bx�	��  �          @��\�e�����A����CE�3�e���������HCV(�                                    Bx�	�p  �          @�=q�^�R����Dz��33CH�3�^�R�&ff�ff��ffCX                                    Bx�	�  �          @��H�X�ÿ�{�J=q�#�CI���X���,(��
=q��=qCZ^�                                    Bx�	ռ  �          @�  �R�\�Ǯ�C�
� �CMff�R�\�3�
��(���ffC\�                                     Bx�	�b  �          @�\)�O\)��Q��@����CO���O\)�9����\)��=qC]�=                                    Bx�	�  �          @����U���  �C�
��CLG��U��0��� ���Ǚ�C[�)                                    Bx�
�  �          @��\�XQ쿡G��Mp��&��CHz��XQ��'
=�  ���HCY��                                    Bx�
T  �          @����;��fff�i���G��CE��;��\)�4z����C\c�                                    Bx�
�  �          @���O\)��G��Y���433CEh��O\)��R�"�\��\)CYh�                                    Bx�
-�  �          @����S33����:�H�(�CQǮ�S33�A녿��H��p�C^�=                                    Bx�
<F  �          @�Q��N{�c�
�W
=�4CC��N{�ff�#�
��CX!H                                    Bx�
J�  �          @�\)�4z�u�j�H�K��CF�=�4z��#33�3�
�{C^#�                                    Bx�
Y�  �          @����&ff�fff�{��\{CG0��&ff�'��Dz��
=Ca:�                                    Bx�
h8  �          @���%�\(��}p��]CFxR�%�%�G
=�!p�Ca{                                    Bx�
v�  T          @�Q���ÿB�\�����hz�CE������"�\�N�R�+G�Cb�                                    Bx�
��  �          @�����Ϳz����\�iQ�CAT{�������Vff�1��C`W
                                    Bx�
�*  �          @�p��>{��=q�X���0�CS�q�>{�L(��
�H���Cc
                                    Bx�
��  �          @���E����k��F�
C<޸�E����C33��RCV}q                                    Bx�
�v  �          @�=q�L��?\)�c33�>
=C*)�L�Ϳ����Z=q�4p�CGxR                                    Bx�
�  �          @���J=q?(���b�\�>z�C(!H�J=q���
�\���8G�CF�                                    Bx�
��  �          @����<(�?��l(��L  C))�<(���Q��b�\�@CJ�                                    Bx�
�h  �          @�{�S33?L���L(��,��C&ff�S33�@  �L���-C@��                                    Bx�
�  �          @�  �Z=q>#�
�R�\�/C1W
�Z=q��{�@  �  CI                                    Bx�
��  �          @����dz�<��L���'=qC3��dzῷ
=�7
=�\)CI��                                    Bx�	Z  �          @����Z=q<��H���*��C3���Z=q��z��3�
�(�CJff                                    Bx�   �          @����@  ?#�
�hQ��G{C(�@  ��=q�aG��?=qCG�
                                    Bx�&�  �          @�G��=p�?   �mp��LffC*z��=p���  �a��?{CJ�)                                    Bx�5L  �          @����.{?^�R�tz��T��C":��.{�u�s33�S33CG\)                                    Bx�C�  �          @�33�0  >��|���[33C*xR�0  �����n�R�J  CN�f                                    Bx�R�  �          @�\)�333��\)�[��6Q�CU���333�N{�p��ۙ�Cd�q                                    Bx�a>  �          @�{�4z��Q��[��0��CY  �4z��\���ff��Cf�3                                    Bx�o�  �          @����0�׿�33�Z=q�6=qCV���0���N�R�(����Ce��                                    Bx�~�  �          @���0�׿�\)�W
=�4�HCV��0���K��
=q����Ce                                    Bx��0  �          @����2�\��
=�P���833CO(��2�\�0  �����  C`�                                    Bx���  �          @��H�0�׿p���b�\�J\)CF�{�0�����/\)��
C]p�                                    Bx��|  �          @��.{�L���n{�R��CDz��.{����>{�=qC]Q�                                    Bx��"  �          @�{�6ff�n{�{��R�CF��6ff�%�E�
=C^T{                                    Bx���  �          @�\)�6ff�fff�~�R�T=qCE���6ff�%�I���33C^8R                                    Bx��n  �          @�Q��7
=�����y���Lp�CL�7
=�:�H�9���\)Ca��                                    Bx��  �          @�Q��0�׿�33�|(��O\)CN���0���AG��9���\)Cc��                                    Bx��  �          @�ff�<(���p��l(��A(�CN� �<(��>{�(���33CaW
                                    Bx�`  �          @�ff�>{���R�j=q�?�CN���>{�>{�'���p�C`��                                    Bx�  �          @��G
=��=q�W
=�+�
CR���G
=�HQ��(���G�Ca@                                     Bx��  �          @��R�HQ�����H����RCW!H�HQ��W�����Q�Cc!H                                    Bx�.R  �          @�
=�P  �'��*�H�G�CZ�)�P  �a녿�p��eG�Cc^�                                    Bx�<�  �          @�33�8Q��7��E��p�C`�\�8Q��{����R����Ci�q                                    Bx�K�  �          @��\�33�HQ��P��� 
=Ci�)�33��  �����G�Cq�                                    Bx�ZD  �          @������@���Z=q�)�RCi��������R��p����
CraH                                    Bx�h�  �          @����(��0���\���,Q�Cdu��(���  ��\)��ffCn�\                                    Bx�w�  �          @���(��   �i���8�HCa�)�(��vff�
�H�ˮCm�
                                    Bx��6  �          @�=q����p��tz��E  C^�����j=q�p���p�ClB�                                    Bx���  �          @�{���K��W
=�!�HCi�������\��\)���
Cq�                                    Bx���  T          @�ff�  �\(��J=q�=qCl�\�  ��\)��=q�m��CsQ�                                    Bx��(  �          @�\)���W��Q��p�Ck������
=��(����\Cr�3                                    Bx���  �          @�\)�  �^{�L���(�Cm{�  ���׿�{�qp�Cs�
                                    Bx��t  �          @���Q��hQ��G
=�
=Co�
�Q���(����H�VffCuG�                                    Bx��  �          @������r�\�C�
���Cq޸����  ����?\)Cv޸                                    Bx���  T          @������p  �Fff��CpQ������\)��z��IG�Cu�                                    Bx��f  T          @�
=���e�E�p�Cn�=�����\��(��X(�Ct�{                                    Bx�
  �          @��R��Q��X���[��%G�Cp8R��Q���G���\)����Cv��                                    Bx��  �          @�����\�r�\�A����Cq����\�������>{Cv��                                    Bx�'X  �          @�녿ٙ������H�؏\Cx�ٙ����þ���5C{O\                                    Bx�5�  �          @�=q���R�y���?\)�	33Cr���R��녿�  �.�\Cw�                                     Bx�D�  �          @�=q��G��e�_\)�$�Cs���G���  ������  Cy�)                                    Bx�SJ  �          @�����|���E��{Cu����zῈ���9��Cyc�                                    Bx�a�  �          @��Ϳ�������G����Cxff�����G�>�@��Cy��                                    Bx�p�  �          @�(���  ���׿�p���Q�Cy�Ϳ�  ����>��H@���Cz�
                                    Bx�<  �          @��\���H��\)��p���ffC|�H���H���
>�  @*=qC~�                                    Bx���  �          @�=q������׿����E��CO\�������?���ADz�CO\                                    Bx���  �          @�녿\���\��(���G�C{���\���>B�\@z�C}�                                    Bx��.  �          @�Q��\�{��<(��	\)Cu�q��\��녿z�H�,��Cy��                                    Bx���  T          @�33�˅����=q����Cz}q�˅���H����3�
C|�f                                    Bx��z  �          @�zῺ�H��{�\)��  C|� ���H��ff���
�aG�C~Y�                                    Bx��   �          @�(���{��������C}��{��ff������Cff                                    Bx���  �          @����
=������  C|�q��
=���R��\)�E�C~��                                    Bx��l  �          @�
=��\)��������(�C{5ÿ�\)��
==�G�?�Q�C|�R                                    Bx�  �          @����G���{�����z�Cys3��G����R�u���C{W
                                    Bx��  T          @��R��������#�
���Cz#׿����{������{C|�                                     Bx� ^  �          @�=q�z�H��z�������
C���z�H���R�L���
=qC���                                    Bx�/  �          @��׿�Q�������H����C5ÿ�Q����
��\)�Dz�C�o\                                    Bx�=�  �          @�G��������������
C}}q��������=q�:=qCO\                                    Bx�LP  �          @�  ��Q���G��z���=qCO\��Q���33�B�\�z�C�l�                                    Bx�Z�  �          @�Q쿹����{����ٮC{�������G����R�X��C}��                                    Bx�i�  �          @�\)��=q������{C}T{��=q��G���33�tz�CB�                                    Bx�xB  �          @�Q쿱���p��   ��
=C|�쿱���녾�
=���C~��                                    Bx���  �          @���������$z���\)C�>�������33����  C��                                    Bx���  �          @�33����������Q�C{
=�����33���
�_\)C}#�                                    Bx��4  �          @�\)�У����
�p��ՙ�Cz�ͿУ���\)���R�O\)C|��                                    Bx���  �          @�  ������(��   ����C{8R������  ��33�g�C}Q�                                    Bx���  �          @��ÿ\��z��%���
C{�f�\������(���C~                                    Bx��&  �          @�녿У���� �����Cz�{�У�������{�`��C|�                                    Bx���  �          @�녿�p���\)� ����G�C|����p����H�����W�C~�                                     Bx��r  �          @��\���H��ff�p���G�Cz����H������z��>�RC|�                                    Bx��  �          @�=q��ff���=q����Cx���ff��Q쾀  �'�C{�                                    Bx�
�  �          @�G����
��=q�%���{Cx�����
��
=�����C{+�                                    Bx�d  �          @�����
=���\�Q���G�Cz�
��
=����=L��>�C|^�                                    Bx�(
  �          @�����(���p���
���RC}Q쿼(����\>��?��C~�=                                    Bx�6�  �          @��׿�p���G���\��(�C|޸��p���녽����HC~u�                                    Bx�EV  �          @��ÿ�������(���z�C{�H������G����
�B�\C}.                                    Bx�S�  �          @�  ��{��G��
=q��
=C{uÿ�{��  �#�
�uC|�q                                    Bx�b�  �          @�\)������33�����\C}T{��������=�Q�?h��C~�)                                    Bx�qH  �          @�{��{��������C~���{��Q�>�  @*�HC�                                    Bx��  T          @�=q���
���Ϳ��
��z�C�
=���
��ff>�p�@���C�`                                     Bx���  �          @�=q�c�
����G�����C�޸�c�
��
=>���@�=qC�&f                                    Bx��:  �          @�=q�#�
���R���
��z�C�LͿ#�
��Q�>Ǯ@��C��H                                    Bx���  �          @��H�����  ��������C��������R?
=@��HC�T{                                    Bx���  �          @��\�������׿����e�C�LͿ������
?Q�AffC�n                                    Bx��,  �          @�����=q���׿���a�C��=��=q���
?Y��Az�C�                                    Bx���  �          @��ÿ0����=q����c33C��0�����?\(�A�\C�,�                                    Bx��x  �          @��R���������Q�����C}z῰����{>���@j�HC~c�                                    Bx��  �          @���ff��{���H���Cw�R��ff��33    �#�
Cy�                                    Bx��  �          @�p���  ���R��33�N�RCy�f��  ����?W
=A��Cy��                                    Bx�j  �          @�zῸQ���=q�z�H�0(�C}Y���Q����?��A:�HC}O\                                    Bx�!  �          @�(����H���Ϳ˅����C|�=���H����>���@���C}^�                                    Bx�/�  �          @�����
���ÿ����W
=C|33���
���?O\)A��C|z�                                    Bx�>\  T          @�{����\)�^�R�\)Cw�׿���{?��AA�Cw�3                                    Bx�M  �          @�ff�G���\)�:�H� ��Cv�)�G���(�?�(�AX��Cvh�                                    Bx�[�  �          @��R��(���  �Q����Cw� ��(���{?��AI�Cw5�                                    Bx�jN  �          @��R�G�����@  �(�Cv�R�G�����?���AT��Cv�\                                    Bx�x�  �          @�{��p����H�(�����
CzaH��p����R?��Aj=qCy�)                                    Bx���  �          @�{�
=���@  �Cu���
=��33?�z�AN�\CuW
                                    Bx��@  �          @�{����z�L����HCt�\�����\?��AB�\Ct�                                    Bx���  �          @�������׿G��	�Cr(�������R?���A?33Cq�
                                    Bx���  �          @������33�@  �{Cs�
�������?�\)AG\)CsxR                                    Bx��2  �          @�{�   ����(����=qCp���   ��z�?�z�AN�RCpW
                                    Bx���  �          @��33��녿Y���\)Cs5��33����?�G�A3\)Cs
=                                    Bx��~  �          @�p��p����\�c�
��\CtO\�p����?z�HA.�RCt8R                                    Bx��$  �          @��H��(���(��(����Cv�R��(�����?��HA\(�Cvu�                                    Bx���  T          @���
=��\)��=q�H��Cp���
=���?(��@��
CqL�                                    Bx�p  T          @��R�Mp��o\)����(�CeY��Mp���33��Q�p��Cg��                                    Bx�  �          @�33�U��o\)������CdT{�U�����\)�AG�Cgu�                                    Bx�(�  �          @�z��g
=�dz������\)C`��g
=��  ��z��A�Cc��                                    Bx�7b  �          @��
�o\)�hQ쿨���dQ�C`#��o\)�w
=>B�\?�p�Ca�f                                    Bx�F  �          @�(���  �XQ쿝p��S\)C\(���  �e>.{?���C]��                                    Bx�T�  �          @�(��y���Y����z��s�C]��y���k�<��
>8Q�C_^�                                    Bx�cT  �          @�p��z�H�\�Ϳ�
=�u��C]T{�z�H�o\)<#�
=��
C_�)                                    Bx�q�  �          @�{�w��^{�˅���C]ٚ�w��s�
����\)C`��                                    Bx���  �          @���z=q�G��   ��ffCZ�\�z=q�i���&ff���C_                                      Bx��F  �          @��R�p���333�����(�CX�f�p���\�Ϳ��\�4  C^��                                    Bx���  �          @�33�Z=q�n�R���R���Cc�)�Z=q����=u?�RCe�3                                    Bx���  �          @����Q��|�Ϳ�Q��Q��CfB��Q���33>��@�p�CgY�                                    Bx��8  �          @��N�R��{��z��F{ChW
�N�R��=q?�@��Ci(�                                    Bx���  �          @�G��O\)���
���
�,Q�Ciu��O\)��?5@�33Ci�)                                    Bx�؄  �          @�
=�E����h���G�Cj���E��p�?W
=A��Ck
=                                    Bx��*  �          @�\)�>�R����
=�F�RCl)�>�R����?��@�  ClǮ                                    Bx���  �          @�=q�{���Ϳ0����p�Cq���{���?���AD  Cq��                                    Bx�v  �          @�ff�4z���33�B�\� (�Cn���4z�����?��A/�Cn5�                                    Bx�  �          @�=q�=p���녿c�
��Ck� �=p���=q?O\)A  Ck�
                                    Bx�!�  �          @�33�3�
��p��W
=�33Cl��3�
��?L��A  Cl�                                    Bx�0h  �          @��\��(����
��=q�E�Cv����(���z�?��RA�Cu��                                    Bx�?  �          @�ff��\��33��G���ffCx�q��\����?�\)A�  Cw�                                    Bx�M�  �          @��׿�Q����׾.{��z�C}(���Q����?��A��C|&f                                    Bx�\Z  �          @�  �����Q�8Q��
=qC�E�����\)?�{A���C�q                                    Bx�k   �          @��\=#�
����>#�
?�
=C�C�=#�
���
?�
=A���C�J=                                    Bx�y�  �          @�(�<����H��{���C�'�<����
?��RA���C�(�                                    Bx��L  �          @�=q�#�
���׾�������C��
�#�
���\?�z�A��RC��
                                    Bx���  �          @�����p�����#�
����C���p����H?ǮA��HC~8R                                    Bx���  �          @��\����p������z�CqE����=q?��
AG\)Cp��                                    Bx��>  �          @���ff���
>\)?˅CtT{�ff��Q�?޸RA��\CrY�                                    Bx���  �          @�녿�33���\    �L��C}���33��Q�?�Q�A�33C{��                                    Bx�ъ  T          @�Q쿼(�����\)�ǮC|�q��(���ff?�\)A��\C{��                                    Bx��0  �          @��H�\��33�L�Ϳ�\C|�׿\����?޸RA��\C{z�                                    Bx���  �          @�z´p�����>\)?�\)C}#׿�p���  ?�Q�A��C{�                                     Bx��|  �          @�
=��(����>�z�@Y��C�q��(����
@�\A��C~\)                                    Bx�"  �          @��Ϳ�{��{>�33@�=qC}�)��{���@�\A��C|=q                                    Bx��  �          @�z῵����>�p�@�33C|�῵��ff@�\A�(�C{=q                                    Bx�)n  �          @��H������=q>�ff@���C{���������\@�A�  Cy�R                                    Bx�8  �          @�=q�33����?�G�A7�Crff�33�n�R@"�\A���CnY�                                    Bx�F�  �          @�Q��33��ff?z�HA5G�Cu8R�33�r�\@!�A�Cq��                                    Bx�U`  �          @�33����Q�?\(�Az�Ct0����x��@(�A�Cp��                                    Bx�d  �          @�(��Q����H?J=qA�HCu��Q��\)@��A��
Cq�                                    Bx�r�  �          @���33���R?\)@�(�Cr��33�}p�@��A�(�Co޸                                    Bx��R  �          @����\��  ?333@��Cs���\�|(�@�A�ffCo�f                                    Bx���  �          @���\)��Q�?:�HAffCx� ��\)��@��A�33Cu�{                                    Bx���  �          @�녿W
=���\?��HA^�\C�R�W
=��=q@7�B
z�C�4{                                    Bx��D  �          @��ͿO\)����?&ff@�C�(��O\)��
=@z�A�ffC���                                    Bx���  T          @�p��O\)����?!G�@�33C�4{�O\)��Q�@33A�{C��)                                    Bx�ʐ  �          @��������Q�
=q���
C!H�������?�{AY�C~��                                    Bx��6  T          @����=q��G��5���C{����=q��
=?�ffA=C{u�                                    Bx���  �          @��Ϳ�z����׽�Q쿆ffCz�\��z����?�{A��HCy��                                    Bx���  �          @����H��ff<��
>�  C}�����H���
?��
A�p�C|�                                     Bx�(  �          @��G���G�?k�A$Q�CsxR�G��z�H@{A���Co�3                                    Bx��  �          @���У�����?.{@��RC{0��У���
=@�A��
Cx�3                                    Bx�"t  �          @�zῺ�H����?�z�AP��C|�Ὼ�H���@1G�B��Cz0�                                    Bx�1  �          @�(���  ���>���@�ffC�Y���  ����@Q�A�\)C��=                                    Bx�?�  �          @�z῔z����
?�33AN�\C�LͿ�z�����@2�\B�\C~ff                                    Bx�Nf  �          @�{���\���?��
A���C�\���\��Q�@H��B��C��                                    Bx�]  �          @����8Q���(�?��\Aj=qC��8Q����
@9��B��C���                                    Bx�k�  �          @�G������H?�\)A}p�C�H�������@>�RBQ�C�%                                    Bx�zX  �          @�  >�
=��\)?�z�A�  C���>�
=�|(�@>{BffC��                                    Bx���  �          @�Q�>.{��=q?��Ao
=C��>.{��=q@8��B\)C�4{                                    Bx���  �          @�  >���(�?n{A+�
C��3>���Q�@$z�A�{C��                                    Bx��J  �          @���>������H?��Am��C��=>�����33@8��B�C�q                                    Bx���  �          @�\)��G���?�z�A�p�C�O\��G��s�
@K�Bp�C��{                                    Bx�Ö  �          @�  ������?�  A���C�4{���p��@P��B"�
C���                                    Bx��<  �          @��׾����@
=qA�Q�C����^{@e�B6p�C�B�                                    Bx���  �          @�G��W
=���H?�A�(�C����W
=�j�H@S�
B%�C��f                                    Bx��  �          @�z῕��\)?��RA���Ck����l��@<(�BC|}q                                    Bx��.  �          @��������(�?���A�\)Cz@ �����dz�@>�RBp�Cv@                                     Bx��  �          @����ff���@�A�33C|�R��ff�QG�@Z=qB0ffCxW
                                    Bx�z  �          @������vff@.{B�C}uÿ���+�@xQ�BT33Cv�                                    Bx�*   �          @��c�
����@33A�ffC�q�c�
�N�R@g�B<�C~�
                                    Bx�8�  �          @�zᾣ�
���\@
=AυC��׾��
�Vff@]p�B733C�Ff                                    Bx�Gl  �          @�z�333��Q�?�
=A��C��3�333�j�H@G
=B��C���                                    Bx�V  �          @�z�#�
����?�\A��
C��#�
�k�@Mp�B$=qC��=                                    Bx�d�  �          @��ͿTz����?�Q�A���C����Tz��`  @Tz�B*�HC�W
                                    Bx�s^  �          @��
������ff?���A�  C�9������i��@@��Bp�C}��                                    Bx��  �          @��
��ff��\)?��RA�
=C�aH��ff�n{@:=qB�C~(�                                    Bx���  �          @��Ϳ�G���G�?�ffA�Cw�R��G��`��@9��BCss3                                    Bx��P  �          @�(���G�����?�33A���C~  ��G��e�@A�Bp�Cz�H                                    Bx���  �          @�ff�W
=���R?�  Aip�C��ͿW
=����@0  B\)C�3                                    Bx���  �          @�\)�����33?\A�ffC�f����tz�@>{BC}^�                                    Bx��B  �          @��ÿ����
?�\)A���CǮ���s�
@Dz�B�\C|�                                    Bx���  �          @�����z����R?��A�z�C|� ��z��dz�@QG�B!��Cx��                                    Bx��  �          @��׿��R���@(�A�G�Cz
=���R�C�
@j�HB<p�Ct                                    Bx��4  �          @�ff��p���p�@�A��HC}�=��p��H��@g�B<  Cx�)                                    Bx��  �          @��׿=p�����?��
AzffC�^��=p��u@-p�B
�C���                                    Bx��  �          @��\���
�w�?��RA�z�C{�f���
�?\)@J=qB0�CvǮ                                    Bx�#&  �          @�����|�;�=q�n�RCzB����tz�?�ffAfffCy�H                                    Bx�1�  �          @�=q?
=��?\)@�33C���?
=��  ?��RA�\)C�*=                                    Bx�@r  �          @�G�?����?J=qA ��C�Z�?��y��@(�A���C���                                    Bx�O  �          @�z�?��
��?(�A z�C��\?��
�p  ?���AԸRC��3                                    Bx�]�  �          @�{?0�����?5A33C��?0���u@�A��\C��                                    Bx�ld  �          @�ff?���  ?��Ad  C�}q?��j=q@�HB�RC�
=                                    Bx�{
  �          @�ff?(�����׾�G���=qC�k�?(����?z�HAP  C���                                    Bx���  �          @����R�j�H@�RB (�Cw޸���R�*=q@b�\BD��Cp�R                                    Bx��V  �          @��Ϳ��y��?�{A�  Cu!H���J�H@2�\B��CpE                                    Bx���  �          @������}p�?Tz�A%��CpQ�����^�R@z�Aԏ\Cm                                      Bx���  �          @��R�.�R�l��?�  AC�
Ci�{�.�R�K�@��Aأ�CeW
                                    Bx��H  �          @�ff�(��s33?�G�Ay��CmY��(��L(�@�HA��HCh��                                    Bx���  �          @�{�'
=�l��?���Al��Cj���'
=�G�@�A�=qCf�                                    Bx��  �          @�
=� ���e?�33A�p�Ck{� ���7�@.{B(�Cd�
                                    Bx��:  �          @���#33�o\)?�A���Ck��#33�E@#33B
=Cfz�                                    Bx���  �          @�
=�3�
�S�
?�ffA�(�Ce�H�3�
�#�
@1G�B33C^E                                    Bx��  �          @��R�-p��B�\@�A�ffCd^��-p����@K�B*�HCZL�                                    Bx�,  �          @���p��G
=@ffA�G�Cg���p��(�@N{B1Q�C]Ǯ                                    Bx�*�  �          @�G��/\)�5@�RA�Cb  �/\)��p�@@��B&�CWٚ                                    Bx�9x  �          @�  �!G��=p�@�A�Q�Ce���!G��
=@@��B)C[�H                                    Bx�H  �          @�{�����U@�\A�{Ct������(�@N�RB@{Cm0�                                    Bx�V�  �          @��׿��n{@��A��C|�=���4z�@U�B=�Cwk�                                    Bx�ej  �          @�  ���R�o\)@�A�C{�f���R�6ff@P  B8�Cv�=                                    Bx�t  �          @�
=��{�e@��B�RC|�
��{�)��@Y��BG{CwQ�                                    Bx���  �          @�ff��
=�c�
@Q�B��C{����
=�(Q�@XQ�BF33Cu��                                    Bx��\  �          @�
=�����a�@��B��Cyuÿ����&ff@XQ�BD��Cs�                                    Bx��  �          @�{��=q�^{@�B��Cy���=q�!�@Y��BG�Cr@                                     Bx���  �          @�{��\)�`  @
=B ��Cx�׿�\)�%�@UBC(�Cr
                                    Bx��N  �          @�{��=q�\(�@{B�\Cx����=q�\)@Z�HBJ
=Cq��                                    Bx���  �          @��ͿxQ��g�@z�A��C�xQ��.{@U�BD\)Czp�                                    Bx�ښ  �          @�=q�c�
�dz�@��A��\C��c�
�,(�@P��BD
=C{��                                    Bx��@  �          @�G��O\)�[�@�HB	�C�UÿO\)� ��@W�BOC|�                                    Bx���  �          @�33�W
=�`��@=qB�
C�H��W
=�%@XQ�BL�C|�                                    Bx��  �          @�
=���
�aG�@ ��B	p�C}� ���
�$z�@^�RBMCx=q                                    Bx�2  �          @��\����`  @p�B�C������$z�@Z�HBR�
C�q�                                    Bx�#�  �          @��H��
=�L(�@��B

=CuǮ��
=��\@QG�BI��Cm��                                    Bx�2~  �          @�(��#�
�%�@(�Bp�Ca=q�#�
�ٙ�@G
=B5p�CU��                                    Bx�A$  �          @�{�7���R@ ��B
�CY�{�7�����@E�B0z�CM.                                    Bx�O�  �          @�G��0  ��\@0  B(�C[ٚ�0  ���@Tz�B=p�CN                                      Bx�^p  �          @���Dz��Q�@:�HBG�CP�H�Dz�8Q�@S33B9=qCA+�                                    Bx�m  �          @���J=q��=q@=p�B��CNxR�J=q���@S33B7  C>��                                    Bx�{�  �          @����7���G�@C�
B)33CS� �7��=p�@\��BEffCB}q                                    Bx��b  �          @�=q�*�H��@S33B9�CT��*�H�
=@i��BT�\C@}q                                    Bx��  �          @�=q�7���G�@L��B2=qCO���7����@`  BI�C=@                                     Bx���  �          @�{�'
=��33@S33B@33CP&f�'
=����@c�
BV(�C;G�                                    Bx��T  �          @����R�>{    <�Cf{��R�5�?fffAV�HCd�R                                    Bx���  �          @�\)�Mp��B�\?:�HA=qC_p��Mp��*�H?�{A��HC[�{                                    Bx�Ӡ  �          @�(��
�H�l��>��R@��
Co���
�H�\(�?�z�A�
=Cm�=                                    Bx��F  �          @�
=�
=q�vff?�@�{Cp���
=q�aG�?�33A�z�Cnp�                                    Bx���  �          @�G���������>�  @N{C|
������  ?\A�Q�Cz��                                    Bx���  �          @�녿����(�?�@�
=C�q�����Q�?�{A��RC~�                                     Bx�8  �          @�녿�\)��  ?\(�A-��C|!H��\)�s33@Aڏ\Cz+�                                    Bx��  �          @�=q���R����?�\)Adz�CzB����R�hQ�@�
A�  Cw��                                    Bx�+�  �          @�G�������G�?���A��Cx�{�����]p�@{B��Cu��                                    Bx�:*  �          @���xQ����H?ǮA��\C�W
�xQ��\(�@,��BG�C~J=                                    Bx�H�  �          @��R����a�?�=qA�ffCq�����5�@333B��Cl\)                                    Bx�Wv  �          @�  �	���c�
?�(�A��Cn�\�	���8��@,��B(�CiO\                                    Bx�f  �          @�����c�
?У�A���Cl�����:�H@'
=B�Cgp�                                    Bx�t�  �          @��\�G��g
=?���A�  Cmٚ�G��?\)@$z�B	��Ch�                                    Bx��h  �          @�33�
�H�g�?�A�=qCo�
�H�:�H@4z�B(�CiT{                                    Bx��  �          @��\���g�?��A��Cl!H���C33@��A�Cg}q                                    Bx���  �          @���P  �@   A���CW��P  ��33@&ffB=qCN�                                    Bx��Z  �          @�  �C33��H@�RA���CZc��C33��z�@5B=qCP�                                    Bx��   �          @��R�QG���  @��B�CP.�QG��u@7
=B��CDY�                                    Bx�̦  �          @��J=q�˅@*�HB(�CN���J=q�=p�@@��B+��CAB�                                    Bx��L  �          @�
=�w
=��G�?��A�(�CL���w
=��ff?��A�(�CF�\                                    Bx���  �          @���~�R�Ǯ?�p�A�
=C9���~�R>W
=@ ��A�{C1                                    Bx���  �          @�
=���>��R?�A�C/�
���?Tz�?��A���C(�                                    Bx�>  �          @�G��y��>��@�
A�
=C0B��y��?h��@��A�
=C&��                                    Bx��  �          @�G��w
==#�
@Q�A��C3s3�w
=?8Q�@G�A�z�C)aH                                    Bx�$�  �          @����c�
<��
@.{B�C3�H�c�
?O\)@&ffB��C'=q                                    Bx�30  �          @����/\)�333@�\A��
Ca���/\)�@0��Bz�CY\)                                    Bx�A�  �          @����E�@  A�  ChǮ��33@B�\B+\)C`��                                    Bx�P|  �          @�������Tz�?��HẠ�CjB�����'�@5�B=qCc��                                    Bx�_"  �          @����-p��<(�@Q�A�=qCcaH�-p��p�@8Q�B  C[:�                                    Bx�m�  �          @�p��g
=��ff@�A�CG�H�g
=�#�
@=qB33C>!H                                    Bx�|n  �          @���qG�����@ffA�\)CE���qG��\)@ffA�p�C<^�                                    Bx��  �          @����z�H���?�p�A��
CC���z�H��@p�A��C:�q                                    Bx���  �          @�G���G��c�
?���A�33C@s3��G�����@�A���C8�f                                    Bx��`  �          @�����=q��ff?��AŅC:\)��=q=���?�Q�A�\)C2��                                    Bx��  �          @�(���=q<�?�A�\)C3����=q?   ?���A�G�C-c�                                    Bx�Ŭ  �          @�G���\)<#�
?��A��
C3���\)>��?���A�z�C-�                                    Bx��R  �          @��H���H�.{?�p�A��HC6L����H>�  ?�(�A�C0�3                                    Bx���  �          @����Q켣�
?�{A��C4:���Q�>�G�?�ffA��C.#�                                    Bx��  �          @�����{=�\)?�Q�A�\)C3���{?
=q?���A�  C,�3                                    Bx� D  �          @�����  ��?\A�G�C4ff��  >���?�(�A���C.��                                    Bx��  �          @�p���Q�=�\)?�Q�AyG�C3���Q�>���?�\)Ak33C.�                                    Bx��  �          @���(��u?���A�(�C4�f��(�>�p�?\A�G�C.�\                                    Bx�,6  �          @�G���  ?   ?��RA��
C-W
��  ?c�
?��A���C(#�                                    Bx�:�  �          @����  >�33?�=qA�ffC/O\��  ?E�?�A�p�C)��                                    Bx�I�  �          @�G����=�\)?У�A�  C3����?�\?��A��C-
                                    Bx�X(  �          @�
=��  �   ?��A�z�C;���  =#�
?���A��
C3aH                                    Bx�f�  �          @�����G��!G�?�Q�A�G�C<�)��G���\)@�\A�Q�C4�R                                    Bx�ut  �          @�
=�w
=���
@   AԸRCB�3�w
=��G�@��A�RC:z�                                    Bx��  �          @��R�w��z�H?�p�A��HCB!H�w��Ǯ@
�HA��C9Ǯ                                    Bx���  �          @�G��w��+�@�A�C=�)�w����
@Q�A��HC4E                                    Bx��f  �          @����w��333@�RA���C>8R�w��u@A�\)C4�
                                    Bx��  �          @����z=q�k�@AۅCA33�z=q���R@��A�G�C8�=                                    Bx���  �          @�  �s33�n{@
�HA�Q�CA� �s33����@A���C8�                                    Bx��X  T          @�ff�k��fff@�A�z�CA�=�k��k�@�RB��C7��                                    Bx���  �          @���p�׿��@{A��CCk��p�׾Ǯ@�HBffC9�R                                    Bx��  �          @����j�H��\)@'
=BffC8^��j�H>�G�@%B
=C-(�                                    Bx��J  �          @�
=�fff���@=qB�CDaH�fff��Q�@'
=B(�C9                                    Bx��  �          @�\)�Z=q��{@#33B
�CI�=�Z=q��R@4z�B�C>W
                                    Bx��  �          @���g���(�@
=A���CF���g��
=q@&ffB�C<}q                                    Bx�%<  �          @�
=�g���{@Q�B �CE��g���(�@%B��C:�q                                    Bx�3�  �          @���h�ÿ�\)@�A�33CE\�h�þ�G�@%�B�C:�
                                    Bx�B�  �          @���l�Ϳ�=q@
�HA�CG�q�l�Ϳ0��@��BQ�C>�)                                    Bx�Q.  �          @�
=�g
=�Ǯ@�\A�CKQ��g
=�s33@��B\)CB�                                     Bx�_�  �          @����mp���
?˅A�ffCQ\�mp��Ǯ@�A��CJ��                                    Bx�nz  �          @�\)�S�
�
=q?�Q�AиRCU+��S�
����@(�B�CMQ�                                    Bx�}   �          @���J�H�$z�?���A�=qC[�J�H���R@(�BffCT)                                    Bx���  �          @��R�@���B�\?�z�As\)CaG��@���'�?�
=AΣ�C]                                    Bx��l  �          @�������~�R>Ǯ@�=qCu������n�R?�
=A�{Cs��                                    Bx��  �          @��%��N{?���Ao�
CgY��%��3�
?�A��Cc�                                     Bx���  �          @����P���!�?�ffA���CY�{�P�׿��H@=qBQ�CS�                                    Bx��^  �          @�33�QG��"�\?��RA�G�CY�)�QG���@%B	\)CRxR                                    Bx��  �          @��\�J�H�{@	��A�G�CY�)�J�H��@.�RB��CQ�3                                    Bx��  �          @���R�\�Q�@z�A��
CW���R�\��  @'�B�\CP�                                    Bx��P  �          @����^�R��@�
A؏\CRG��^�R��@!G�Bz�CJ(�                                    Bx� �  �          @�  �fff����@�A�\)CK��fff�xQ�@!G�B33CC�                                    Bx��  �          @����l�Ϳ�ff?�{A��CN  �l�Ϳ�  @G�A�CF�                                    Bx�B  �          @�Q��l�����?���A���CR��l�Ϳ��H?��A�G�CL�                                     Bx�,�  �          @�ff�,���Z�H?:�HA\)Cg���,���G
=?���A�\)Cd�q                                    Bx�;�  �          @��R�R�\�.�R?�z�As�C[���R�\��?�A�  CWW
                                    Bx�J4  �          @�p��?\)�=p�?�ffA��HC`���?\)�!G�@G�A��HC\�                                    Bx�X�  �          @���J�H�p�?���AÙ�CYǮ�J�H��33@��B
=CR�f                                    Bx�g�  �          @�{�333�8��?�=qAÅCa���333�z�@!G�B
�HC[�{                                    Bx�v&  �          @�z��3�
�333?�\A�C`���3�
�  @�B(�CZ�H                                    Bx���  �          @�z��;��*�H?���A�(�C^W
�;��ff@�RB
{CW�)                                    Bx��r  �          @�z��AG��#�
?���A�=qC\W
�AG��   @��B\)CU}q                                    Bx��  �          @�p��R�\�6ff?�RA�RC\�)�R�\�%?��A��CZ.                                    Bx���  �          @���g��!G�?W
=A-�CV�H�g��{?��HA�ffCS��                                    Bx��d  �          @����c33�\)?�Q�A�G�CT=q�c33��p�@p�A�CM��                                    Bx��
  �          @����U���
?�(�A��CV��U���(�@\)Bp�COT{                                    Bx�ܰ  �          @�G��U�{@�A�p�CU���U��{@%�BQ�CM��                                    Bx��V  �          @���(Q��N{?��A�
=Cf�R�(Q��(Q�@)��B�RC`�3                                    Bx���  �          @��\��H�S33?�
=A�Q�Ci���H�,(�@-p�BffCd)                                    Bx��  �          @����\)�Q�?�=qA�p�Ch�q�\)�,��@&ffBG�CcL�                                    Bx�H  T          @�33�333�H��?���A�Q�CdJ=�333�#�
@%B	p�C^z�                                    Bx�%�  T          @�33�/\)�{@,(�B(�C^
=�/\)���H@N�RB3��CS��                                    Bx�4�  �          @��H���i��?�
=A�ffCp�����B�\@2�\B��ClL�                                    Bx�C:  �          @��H����s33?�  A���Cs�����N�R@*=qB�RCo��                                    Bx�Q�  �          @�=q�z��j=q?�A�{Cp��z��E�@*�HB(�Cl�                                    Bx�`�  �          @����{�c�
?�33A�(�Cn
=�{�A�@   BCi�R                                    Bx�o,  �          @���(Q��`��?�p�AzffCi)�(Q��E�@�A��HCe��                                    Bx�}�  �          @�G��6ff�P��?��A�\)Cd�\�6ff�4z�@�A���C`�H                                    Bx��x  �          @�G��>{�HQ�?�\)A��Cb���>{�+�@�A�
=C^#�                                    Bx��  �          @��H�E�1�?�AƏ\C^��E�p�@#�
B  CW�)                                    Bx���  �          @���Mp��   @33A�  CY�3�Mp���z�@'
=B
=CR��                                    Bx��j  �          @���L(��\)@�
A�  CY���L(����@'�B��CR��                                    Bx��  T          @��\�g���=q@�
A���CN�=�g���G�@��BG�CG)                                    Bx�ն  �          @��\�^{��
@	��A�z�CR��^{���H@&ffB
z�CJ�                                     Bx��\  �          @�  �e���R?�=qA��CQ��e��(�@G�A�ffCJW
                                    Bx��  �          @����l�Ϳ��?��A�Q�CO{�l�Ϳ���@��A�G�CH��                                    Bx��  �          @�Q��l�Ϳ�\)?�G�A�(�CN�\�l�Ϳ�\)@
�HA��CHW
                                    Bx�N  �          @�  �j=q��?�\)A�ffCNW
�j=q���@��A�p�CGn                                    Bx��  �          @�  �`  ���H@G�A��CQ.�`  ���@(�B�
CI�                                    Bx�-�  �          @�\)�n{��\)?��A�CN���n{��z�@33A�Q�CH��                                    Bx�<@  �          @�  �w
=��?�{A�G�CK\)�w
=��(�?�(�Aϙ�CE�                                     Bx�J�  �          @�Q��q녿ٙ�?�  A�\)CL+��q녿��H@�A��CE�                                     Bx�Y�  �          @�  �p  ��p�?�A��
CL�\�p  ��p�@(�A�  CF+�                                    Bx�h2  �          @�p��L�Ϳٙ�@!G�B\)CO��L�Ϳ��\@7
=B!�RCE�R                                    Bx�v�  �          @�{�Tz��z�@z�A���CS�R�Tz��  @ ��B
z�CLJ=                                    Bx��~  �          @�
=�X����?�{A��RCW�H�X�ÿ��H@	��A���CR                                    Bx��$  �          @�  �aG��33?���A��RCU0��aG���@�A��\CO��                                    Bx���  �          @�  �e���R?�ffA��CS޸�e����
@33A��CNff                                    Bx��p  �          @�  �j=q�
=q?�(�A���CR�
�j=q�޸R?��HA�(�CMc�                                    Bx��  �          @�Q��p  �   ?��RA�
=CP#��p  ��=q?�Q�A�{CJ��                                    Bx�μ  �          @����s33���R?�A���CO���s33��=q?�\)A��HCJ�{                                    Bx��b  �          @�  �w����?��
A�Q�CM���w��\?ٙ�A�  CIff                                    Bx��  �          @�\)�~�R���H?�z�Ao�CK5��~�R��\)?��A���CG�                                    Bx���  �          @��R�~{��
=?��Am�CK��~{����?\A�p�CF�)                                    Bx� 	T  �          @�z��w���G�?�{AiG�CLs3�w���Q�?�  A�CHaH                                    Bx� �  �          @����y����p�?���Ag33CK�3�y����z�?��RA��
CG�                                    Bx� &�  �          @���w��ٙ�?��\A�  CK���w����?��A��CG�                                    Bx� 5F  �          @���|(�����?�
=Axz�CI��|(����R?\A�{CEz�                                    Bx� C�  �          @���|�Ϳ�  ?�  A�CH�q�|�Ϳ�33?���A�p�CD8R                                    Bx� R�  �          @�  ���׿���?�ffA��
CIG����׿��H?��A�ffCD��                                    Bx� a8  �          @����Q쿾�R?�{A���CHn��Q쿏\)?�
=A�p�CC�H                                    Bx� o�  �          @�  ���ÿ�{?���A��RCF�R���ÿxQ�?޸RA�  CA�)                                    Bx� ~�  �          @�����\��p�?��A�Q�CD�3���\�Y��?��A�  C?��                                    Bx� �*  �          @�  �\)��33?��A�CGY��\)�}p�?�=qA��
CA�                                    Bx� ��  �          @�{�z=q��
=?\A���CH!H�z=q���\?���A�=qCB��                                    Bx� �v  �          @��{���z�?�(�A��RCG�q�{����\?�G�A���CB��                                    Bx� �  �          @����ÿ��?��\A�\)CFc����ÿ�  ?�ffA�Q�CA�H                                    Bx� ��  �          @������ÿ�ff?�Ax  CE�����ÿz�H?���A�Q�CA��                                    Bx� �h  �          @�����H��?�
=Aw�CD����H�\(�?�A��RC?�)                                    Bx� �  �          @���~{���?��HA��RCC�q�~{�Q�?���A�G�C?�)                                    Bx� �  �          @���x�ÿ���?z�HAU��CG&f�x�ÿ�=q?��\A��
CC�                                     Bx�!Z  �          @���|�Ϳ�{?���Ak
=CC��|�ͿTz�?��A��C?�{                                    Bx�!   �          @�\)�x�ÿ���?��Az=qCE5��x�ÿfff?��A��CA�                                    Bx�!�  �          @�\)�z�H���R?}p�AXQ�CE���z�H�xQ�?�  A�(�CA��                                    Bx�!.L  �          @�����G����?W
=A5��CC���G��\(�?���Aip�C@�                                    Bx�!<�  �          @�����G���{?uAO33CCW
��G��Y��?���A���C?ٚ                                    Bx�!K�  �          @�G��{���{?���Ag�
CG
�{�����?�\)A�z�CC.                                    Bx�!Z>  �          @����vff��  ?�\)As\)CIO\�vff��Q�?���A�{CE5�                                    Bx�!h�  �          @���qG���=q?���A�z�CJ�qG���(�?�Q�A��RCE�H                                    Bx�!w�  �          @���a녿�?У�A��
CO�=�a녿�z�@G�A߮CI��                                    Bx�!�0  �          @����W���Q�?�ffA���CQ���W����H@p�A�ffCKs3                                    Bx�!��  �          @�  �W�����?�Q�A�=qCR��W����R@
=A�  CK��                                    Bx�!�|  �          @����e���G�?��A���CN5��e�����?�Aԏ\CH�f                                    Bx�!�"  �          @�{�e��z�?��A�z�CL���e���?޸RA�{CG��                                    Bx�!��  �          @�ff�b�\��ff?�{A�G�CO��b�\��
=?�  AĸRCJ
=                                    Bx�!�n  �          @�Q��h�ÿ�\?�=qA�G�CM�q�h�ÿ�z�?��HA�
=CI5�                                    Bx�!�  �          @����r�\��ff?�  A�(�CJ:��r�\���H?�=qA�p�CE�R                                    Bx�!�  �          @��H�r�\�У�?�{A�=qCKB��r�\��G�?��HA�
=CFp�                                    Bx�!�`  �          @�33�o\)��33?�p�A�\)CK� �o\)��  ?�=qAƸRCF�                                    Bx�"
  �          @����n�R���?�{A�G�CK���n�R���
?ٙ�A��HCF��                                    Bx�"�  �          @�G��mp��˅?��HA�
=CK.�mp�����?��A�G�CE�R                                    Bx�"'R  �          @��H�p  ��\)?�Q�A�CKQ��p  ��p�?��
A�=qCF8R                                    Bx�"5�  �          @��
�tz΅{?�=qA���CG��tz�s33?�{A��CA�                                    Bx�"D�  �          @�G��w���  ?�\)A�p�CE�H�w��c�
?У�A�Q�CA                                      Bx�"SD  �          @�\)�w���G�?�\)A��CB���w��(��?���A�ffC=�                                    Bx�"a�  �          @���w��}p�?�
=A���CBY��w��!G�?�\)A�C==q                                    Bx�"p�  �          @����xQ�aG�?�
=A�{C@���xQ��?˅A�\)C;�R                                    Bx�"6  �          @����l(��У�?��A�(�CK�
�l(���G�?�p�A���CF��                                    Bx�"��  �          @����qG���=q?��
A��CGs3�qG��p��?�ffA���CA�q                                    Bx�"��  �          @�Q��p�׿u?�
=A�G�CBaH�p�׿��?�{A�z�C<J=                                    Bx�"�(  �          @����|(��Ǯ?�{A�
=C9���|(��#�
?�z�A�z�C4�                                    Bx�"��  �          @����g
=�
=?�AzffCRW
�g
=���
?У�A�Q�CNJ=                                    Bx�"�t  �          @��R�u��ff?���A��
CM&f�u��Q�?�G�A���CH}q                                    Bx�"�  �          @�ff��=q�s33?���A�G�CA33��=q�
=?У�A�\)C<G�                                    Bx�"��  �          @�33�����@  ?���A��
C>s3�������?�G�A�
=C9�                                     Bx�"�f  �          @�33��녿333?�33A��RC=Ǯ��녾�Q�?��
A��\C9                                      Bx�#  �          @�(�������ff?��A��HCB�������:�H?��RA��C>+�                                    Bx�#�  �          @�=q�|(���
=?�=qA��CD�)�|(��W
=?���A�ffC?�R                                    Bx�# X  �          @��\�|�Ϳ���?��A��\CD���|�Ϳ^�R?��
A�C@p�                                    Bx�#.�  �          @�(��~�R��ff?��
A�CF��~�R�xQ�?��A�33CA��                                    Bx�#=�  �          @����~�R��  ?���A���CEY��~�R�c�
?У�A�ffC@�H                                    Bx�#LJ  �          @��
�z=q��
=?��A�ffCH{�z=q��=q?У�A��CCz�                                    Bx�#Z�  �          @�{�|(���\)?�p�A��HCJc��|(���ff?���A�  CF5�                                    Bx�#i�  �          @��vff��\?��
A�  CL���vff��
=?�33A�
=CHh�                                    Bx�#x<  �          @����s33��{?���A}G�CN��s33���
?˅A�{CI�                                    Bx�#��  T          @�(��n{���R?�Ax��CP#��n{��?���A�G�CL#�                                    Bx�#��  �          @����n{��
?�33As\)CP�3�n{�޸R?���A�{CM
=                                    Bx�#�.  �          @����fff���?���Ah(�CT��fff����?˅A��CP\)                                    Bx�#��  �          @��dz��
=?�33Aq��CUs3�dz���?�z�A�z�CQ�H                                    Bx�#�z  �          @�z��a��ff?���Ao\)CU�3�a���?��A�CQ�                                    Bx�#�   �          @����a���?�G�AS�CV�
�a����?��A��CS5�                                    Bx�#��  �          @�z��`  �=q?�=qAd(�CV�
�`  �ff?�{A��CR�q                                    Bx�#�l  �          @����`  �=q?�{AiG�CV���`  �ff?У�A�(�CR�f                                    Bx�#�  �          @�{�]p��G�?���A�\)CUQ��]p�����@�Aߙ�CP�                                    Bx�$
�  �          @�(��a��ff?��RA���CR��a녿ٙ�?�
=A��CM��                                    Bx�$^  �          @����N{�!G�?У�A��CZ��N{��@
=qA�G�CT�H                                    Bx�$(  �          @����C�
�1�?\A��C^B��C�
�
=@�A噚CY�H                                    Bx�$6�  �          @��E�8Q�?�=qA��C_��E�   ?���A�33C[�                                    Bx�$EP  �          @��  �e�?�
=Ay��Cmٚ�  �Mp�?��HA��HCk�                                    Bx�$S�  �          @�{�(Q��R�\?�33A���Cgu��(Q��8��@
=A�33Cc�                                     Bx�$b�  �          @�{�"�\�Z�H?�(�A�Q�CiW
�"�\�C33?�(�A�G�Cf(�                                    Bx�$qB  T          @�ff�  �j�H?�\)Ah��CnxR�  �Tz�?�A��Ckٚ                                    Bx�$�  �          @���\�j�H?n{ABffCm�R��\�W
=?޸RA��Ck��                                    Bx�$��  �          @���
�H�n�R?^�RA5��Co�\�
�H�\(�?�Q�A���Cm�R                                    Bx�$�4  �          @�33���R�q�?J=qA'
=Cr5ÿ��R�`  ?У�A�  Cp\)                                    Bx�$��  �          @��
��ff�y��?E�A#\)Cu5ÿ�ff�g�?��A��RCs��                                    Bx�$��  �          @���z��p  ?Q�A-p�Cq��z��]p�?�33A�{Co�                                    Bx�$�&  �          @��
���p  ?fffA>=qCq�
���\��?�p�A��HCo��                                    Bx�$��  �          @�z�����vff?J=qA%G�Cs)�����dz�?��A��
CqT{                                    Bx�$�r  �          @�z�\)���?��@�C�H��\)��=q?�G�A��C�3                                    Bx�$�  �          @��ͿO\)����?�\@ָRC�� �O\)����?�(�A��HC�T{                                    Bx�%�  �          @���E���G�?z�@�(�C��ͿE�����?��A�z�C��                                     Bx�%d  �          @�(��+���\)?\(�A5��C��׿+��{�?�ffA�z�C�+�                                    Bx�%!
  �          @����(����>Ǯ@�33C}�)��(��~{?���A�p�C|�
                                    Bx�%/�  �          @�(���{��z�>���@���C{�H��{�|(�?�=qA��
C{                                    Bx�%>V  �          @��
��  ��p�>\@�
=C}T{��  �~�R?���A�=qC|�\                                    Bx�%L�  �          @�zΰ�
��>�p�@��C|�ÿ��
�~�R?��A��C|.                                    Bx�%[�  �          @�{���
���>�=q@\��C}&f���
��=q?�(�A�
C|z�                                    Bx�%jH  �          @�{�����Q�>�@�{C쿑���G�?�z�A���C~O\                                    Bx�%x�  �          @��Ϳ�����R?+�A�C�{����|��?�{A�\)C~�                                    Bx�%��  �          @��Ϳh����Q�?�\@��
C�  �h������?��HA��C���                                    Bx�%�:  �          @�p��k���(�����G�C����k���G�?aG�A9G�C�z�                                    Bx�%��  �          @��ͽL����(����R����C��=�L�����\?0��Ap�C��=                                    Bx�%��  �          @�(��k������G����RC����k�����?aG�A9C�z�                                    Bx�%�,  �          @�(��z�H��  >�p�@��\C�y��z�H���?���A���C�,�                                    Bx�%��  �          @�=q�������<�>ǮC��f������?�  AV�RC�Ф                                    Bx�%�x  �          @�33�����G�>��
@��RC�^�������?��
A��C�4{                                    Bx�%�  �          @����\)���?�@�z�C~�f��\)�|(�?���A�(�C~�                                    Bx�%��  T          @��Ϳ�������?+�A  Cx�q�����r�\?���A�\)Cwp�                                    Bx�&j  �          @�33����z�H?fffA?\)CwG�����g
=?�G�A��
Cu�{                                    Bx�&  �          @�G���  �z�H?W
=A5G�Cy!H��  �hQ�?��HA�=qCw�)                                    Bx�&(�  �          @���
=�U?��A��HCj�q�
=�<��@�
A���CgT{                                    Bx�&7\  �          @�=q��(��g�?��HA���Cqn��(��P  @   A��
Cn�                                    Bx�&F  
�          @�=q�4z��(�@33A�z�C\޸�4z��z�@#33BQ�CV!H                                    Bx�&T�  
�          @��
�N{�G�@	��A�CT#��N{��p�@#33B  CL��                                    Bx�&cN  
�          @��\�E�
=q@��A�\CV��E��{@$z�B��CO�H                                    Bx�&q�  �          @�(��Q��B�\?�
=A���Ck
=�Q��%�@z�B=qCf��                                    Bx�&��  T          @��\�I������?��HAƣ�CS���I����G�@
=A�G�CM�                                    Bx�&�@  T          @����2�\��G�@��BCTJ=�2�\��p�@�RB�\CK�{                                    Bx�&��  
�          @��׿�p��{?ǮA��ClE��p���@�B#��Cf�f                                    Bx�&��  �          @�(�?����q녾W
=�A�C���?����n�R?&ffA�C���                                    Bx�&�2  �          @���?B�\��ff�������RC�q?B�\��p�?!G�A33C�'�                                    Bx�&��  �          @��H?!G����׾������\C�4{?!G���  ?z�@���C�9�                                    Bx�&�~  �          @��\?8Q���\)�����C�Ф?8Q����>�G�@�(�C��                                    Bx�&�$  �          @��\?(����  ��G�����C�aH?(�����?
=q@�ffC�c�                                    Bx�&��  �          @��
?(�������Ǯ���C�^�?(������?
=@��C�e                                    Bx�'p  T          @�(�?+���녾�����Q�C�o\?+���G�?
=@��RC�t{                                    Bx�'  T          @�z�?����H�\����C�h�?����?(�A (�C�n                                    Bx�'!�  �          @�z�?
=q���\�������C���?
=q���?z�@�z�C���                                    Bx�'0b  �          @��?
=����������  C���?
=����?
=@���C���                                    Bx�'?  �          @��H?@  ��Q쾽p���C��?@  ��\)?��A (�C��                                    Bx�'M�  �          @�33?&ff���þ�p���p�C�P�?&ff��Q�?(�A�C�W
                                    Bx�'\T  T          @��
?���녾�������C���?���G�?
=@��C���                                    Bx�'j�  T          @��
?(���녾Ǯ���C�?(���G�?��@��C�
=                                    Bx�'y�  �          @��?���녾Ǯ��C�t{?���G�?��@�z�C�y�                                    Bx�'�F  �          @�p�?5��33��33���HC���?5���?&ffAffC���                                    Bx�'��  �          @�p�?333��33�����vffC��?333����?333AffC���                                    Bx�'��  �          @�{?����(���������C��?�����H?+�A�
C��                                    Bx�'�8  �          @�?�R���
��Q���z�C�
=?�R���H?&ffAffC��                                    Bx�'��  	f          @�>B�\��p��.{�\)C�>�>B�\���H?W
=A.�RC�E                                    Bx�'ф  �          @�z�>\)��z�#�
�z�C��>\)����?uAI��C���                                    Bx�'�*  	�          @�z�=�����z����z�C�� =�������?aG�A8Q�C���                                    Bx�'��  T          @�zὣ�
��������C�y����
���H?
=@�Q�C�xR                                    Bx�'�v  "          @��=�����zᾏ\)�g
=C���=������H?:�HAp�C���                                    Bx�(  
�          @��<#�
���;#�
�Q�C�R<#�
��=q?Y��A1G�C��                                    Bx�(�  "          @�p�=L����p����\C�P�=L�����\?fffA;33C�Q�                                    Bx�()h  
�          @�=�G���p�<#�
>�C��q=�G�����?��\AU��C�                                    Bx�(8  N          @�p�>aG����>L��@%�C�k�>aG����?���A|��C�y�                                    Bx�(F�  �          @��>�\)���ͽ�G�����C��\>�\)���?fffA=�C�ٚ                                    Bx�(UZ  
�          @���>aG���(�>�\)@n{C�n>aG���ff?��
A��C�}q                                    Bx�(d   �          @��>�  ��z὏\)�fffC���>�  ��G�?p��AE�C���                                    Bx�(r�  T          @�>�������u�:�HC��
>������?uAG�
C��                                    Bx�(�L  "          @��>��R���R�\��=qC��R>��R��?(��A�\C��)                                    Bx�(��  "          @�  >�(���
=�����w
=C���>�(���p�?:�HAp�C��H                                    Bx�(��  "          @���>�
=���þu�?\)C��>�
=���R?O\)A#�C��\                                    Bx�(�>  
Z          @��\>�33��녾k��8��C�1�>�33���?Q�A%��C�9�                                    Bx�(��  
�          @��H>�{��=q�8Q��p�C��>�{���?aG�A0��C�(�                                    Bx�(ʊ  �          @�33>�Q����\�#�
�G�C�9�>�Q���  ?fffA3�
C�C�                                    Bx�(�0  "          @��\>W
=��녾.{�
=C�T{>W
=��\)?c�
A2�HC�Z�                                    Bx�(��  
�          @��\�����녾��
��  C�` �����Q�?=p�A��C�Z�                                    Bx�(�|  �          @��k������\)�_\)C���k���33?L��A�C���                                    Bx�)"  "          @��R��z���ff�u�8Q�C�B���z����
?Y��A&�RC�<)                                    Bx�)�  "          @��;������
�Ǯ����C��þ������\?0��A��C���                                    Bx�)"n  �          @��������;Ǯ��Q�C�  �������?333A��C�)                                    Bx�)1  �          @�녾���Q�������C������\)?(��A��C���                                    Bx�)?�  
�          @�=q�   ���þ�����ffC���   ��\)?8Q�Ap�C��3                                    Bx�)N`  �          @��H����G���z��e�C�������?E�A\)C���                                    Bx�)]  
�          @���
=�����\)�Y��C�` �
=����?L��A{C�S3                                    Bx�)k�  
�          @��Ϳ(�����B�\�
=C�7
�(�����?aG�A.�RC�%                                    Bx�)zR  
�          @��
�\)���\��Q쿓33C���\)��\)?xQ�AB�\C�q�                                    Bx�)��  
�          @��������\�L�Ϳ.{C��R������R?�G�AJ=qC���                                    Bx�)��  "          @���\����<#�
>��C��)��\����?��AX  C��                                    Bx�)�D  	�          @�ff��\��p����ǮC��q��\����?��AO�C�Ǯ                                    Bx�)��  
�          @�
=�   ��{<#�
>��C��R�   ���?���AXz�C��                                    Bx�)Ð  "          @��
��\���\>.{@�
C��\��\��p�?�p�Av�HC���                                    Bx�)�6  (          @����B�\��\)>�=q@VffC�5ÿB�\��G�?�ffA�
=C��                                    Bx�)��  �          @�녿E���\)>���@mp�C�#׿E���G�?�=qA�  C���                                    Bx�)�  �          @��H�z�����=��
?�G�C�XR�z�����?��Af=qC�9�                                    Bx�)�(  T          @�z�������
�#�
����C��H�������?���AUC�o\                                    Bx�*�  T          @��������33=��
?uC��;�����ff?�33Ag
=C���                                    Bx�*t  �          @����\)��33��Q쿔z�C�C׾�\)���?}p�AEp�C�9�                                    Bx�**  "          @������33��=q�Z=qC��
����G�?O\)A"{C��
                                    