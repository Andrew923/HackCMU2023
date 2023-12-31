CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230406000000_e20230406235959_p20230407021718_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-07T02:17:18.488Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-06T00:00:00.000Z   time_coverage_end         2023-04-06T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxu=��  T          @�@�G��}p��L���/\)C��@�G��=p��0�����C���                                    Bxu=�f  T          @�ff@}p����;.{�C���@}p�����Y���<Q�C�|)                                   Bxu=�  
�          @�\)@z�H��  �.{��
C���@z�H��Q�n{�K�
C��R                                   Bxu=��  �          @���@h���Q�L���/\)C��R@h�ÿ��H�����C�n                                    Bxu>	X  
�          @���@H���<(�����ffC�s3@H���\)�Ǯ���RC��\                                    Bxu>�  T          @���@~{��(����Ϳ��C��R@~{�����\(��:�\C���                                    Bxu>&�  �          @�=q@W��&ff�(���RC�5�@W���녿����
=C�W
                                    Bxu>5J  �          @���@c�
���#�
��C�+�@c�
��Q쿯\)���C��
                                    Bxu>C�  �          @�  @g��
�H���Ϳ���C��=@g���ff��(���{C���                                    Bxu>R�  �          @��@o\)��ý���{C�
@o\)��G���(����C�]q                                    Bxu>a<  
�          @��H@s33�ff�#�
��C��=@s33��\�����q�C���                                    Bxu>o�  �          @��\@l(����<#�
>B�\C�J=@l(���
=��z��y��C�0�                                    Bxu>~�  �          @�33@^{�'��.{�
�HC�s3@^{�	����G���=qC�
                                    Bxu>�.  "          @��
@dz��   �.{�  C��@dz���\������
=C��                                    Bxu>��  �          @�33@i���
=�L�Ϳ&ffC��@i�����R���\��Q�C���                                    Bxu>�z  T          @���>�Q��c33?�A�33C��>�Q��h�ÿY���Q�C�ٚ                                    Bxu>�   
�          @�=q�
=����?�A�  C���
=���aG��<��C�                                      Bxu>��  
�          @�G��#�
��33?�p�A�  C��ü#�
��zῊ=q�ip�C���                                    Bxu>�l  
�          @��׽L����p�?J=qA*�HC����L���~{��  ��33C���                                    Bxu>�  �          @�녿W
=��33?@  A!C�9��W
=�x�ÿ��R���\C��                                    Bxu>�  
�          @��������H?�{As�C������=q����ffC�\                                    Bxu?^  )          @��>�p���=q?z�@��
C�j=>�p��\)��  ��ffC��q                                    Bxu?  
�          @�33?�ff��=q?fffABffC�=q?�ff�|(������33C�t{                                    Bxu?�  
�          @���?���vff?��HA��
C�
?�����
=q��C��f                                    Bxu?.P  "          @�33>�ff�u�?�z�A���C�T{>�ff���׾�33���RC��q                                    Bxu?<�  "          @�33>������?��A�z�C�޸>������c�
�AC��                                    Bxu?K�  �          @�33���
��z�?��\A]��C��3���
��=q���\��(�C��=                                    Bxu?ZB  
�          @�33?aG���33    =uC��?aG��`�������HC��                                    Bxu?h�  
�          @�\)?�  ���H>�(�@���C��?�  �n{��  ��ffC��3                                    Bxu?w�  
�          @�Q�@:�H�@��?��A���C�{@:�H�[��#�
��p�C�>�                                    Bxu?�4  �          @�Q�@,(��Z=q?���Ah��C�  @,(��`  �L���$(�C�                                    Bxu?��  T          @�G�@p��n{?:�HA��C��3@p��dzῦff����C�Ff                                    Bxu?��  T          @�G�@�vff���
�s33C���@�P����
���HC���                                    Bxu?�&  T          @��@
�H�~�R�#�
��G�C�N@
�H�X�������C�Q�                                    Bxu?��  M          @�G�@���n�R�Q��'
=C�K�@���1G��(Q����C�XR                                    Bxu?�r  
�          @���@�
���׾�z��k�C��\@�
�R�\�z���
=C�H                                    Bxu?�  �          @��?�Q���  �8Q����C��
?�Q��c33�ff��=qC���                                    Bxu?�  
�          @�(�@   ���\�E��G�C��@   �Fff�0����C�q�                                    Bxu?�d  T          @�p�?�  �����\�K33C�XR?�  �C�
�A��"��C��H                                    Bxu@

  "          @��?�p���ff�u�>=qC�0�?�p��G
=�>�R��C���                                    Bxu@�  T          @�p�?�{�������\�~�HC�Z�?�{�4z��K��,�C��q                                    Bxu@'V  "          @�?�{���H�#�
��33C�,�?�{�X���1��  C���                                    Bxu@5�  �          @��?�\)��{�����q�C�f?�\)�\(��=q��\)C�Ff                                    Bxu@D�  
(          @��?����������Mp�C���?����\(��
=����C��H                                    Bxu@SH  �          @���?k���G�>���@p  C��{?k������z���  C�`                                     Bxu@a�  �          @��>�p�����?\(�A+33C�XR>�p���=q��=q����C�t{                                    Bxu@p�  T          @�33=u����>�ff@�p�C�]q=u��z������p�C�ff                                    Bxu@:  �          @��?W
=��ff?��@���C�]q?W
=��33�����C��\                                    Bxu@��  
�          @�33>�G���  ?!G�@�z�C�>�G���{��\��Q�C��
                                    Bxu@��  
�          @��?5���H��=q�S�
C�h�?5�s33�%����C�L�                                    Bxu@�,  
�          @�{?+���=q�=p���C�/\?+��c33�>{���C�]q                                    Bxu@��  �          @�ff>�����G������R�RC��H>����W
=�O\)�/ffC��=                                    Bxu@�x  �          @�p�?������
���
��\)C�C�?����0���Z�H�?�C�K�                                    Bxu@�  "          @���?��
��Q�xQ��B{C���?��
�J�H�@���"\)C��=                                    Bxu@��  
�          @�(�?�����z�5�33C�H?����Z=q�6ff�33C��f                                    Bxu@�j  
�          @�z�?s33���׾����k�C���?s33�n�R�#�
�z�C��                                    BxuA  "          @��=�G����?�=qA_\)C���=�G���녿�����C��)                                    BxuA�  "          @�p���  �p�@n�RBSCg�f��  �s33@�A�CuO\                                    BxuA \  T          @����z��(�@j�HBT��Ch�
��z��p��@�\AӅCv#�                                    BxuA/  T          @�(����
�0��@S�
B9=qCp���
���?�Q�A��HCyL�                                    BxuA=�  
(          @�(���G��\(�@(Q�B=qCvQ��G�����?�@�  Cz��                                    BxuALN  
�          @�=q�s33�p��@A���C޸�s33��=���?��
C��
                                    BxuAZ�  �          @�G�����I��@FffB0�
C��)�����G�?���A^�\C���                                    BxuAi�  
�          @�G�>aG��,��@C�
BB=qC�S3>aG��xQ�?��\A��HC���                                    BxuAx@  "          @��H?(���@tz�Bb��C�
=?(���G�@�
A�=qC�K�                                    BxuA��  �          @�33=����Q�@n�RBe�RC�4{=����|(�@ ��A��
C���                                    BxuA��  
�          @��
��=q�G�@1G�BCv�H��=q���H?L��A)�C{��                                    BxuA�2  T          @���p��tz�?���A�p�Co���p�����\)�ٙ�Cq��                                    BxuA��  
�          @�ff�����}p�@   A���CxJ=������p���\)�Z�HCzff                                    BxuA�~  �          @�ff���\����@	��A���C����\��녾8Q��\)C���                                    BxuA�$  "          @�Q쿆ff���?�
=A��C�=��ff��=q��G����HC��H                                    BxuA��  "          @��׿Y�����R?�z�A�  C��H�Y����=q����I��C���                                    BxuA�p  	�          @���n{��ff?��A��C��q�n{��G��}p��K33C��)                                    BxuA�  �          @��
��\)�>�R@HQ�B*��Cq���\)����?���Aqp�Cx��                                    BxuB
�  	�          @�(���(��@  @EB&��Cp=q��(�����?�33Af=qCw�                                    BxuBb  �          @�z�޸R�5@N�RB0Q�Cn���޸R���\?�{A�  Cv�f                                    BxuB(  �          @����\)�#33@U�B9��Ci�R��\)�xQ�?˅A�=qCt=q                                    BxuB6�  "          @�  ��33�H��@5B!G�Cy�쿓33����?aG�A7�
C~n                                    BxuBET  T          @��R>��p  @
=qA��
C���>����\�#�
�B�\C��R                                    BxuBS�  T          @���?�
=�p��@�A�\)C��{?�
=��33�#�
���
C��
                                    BxuBb�  
�          @��?�\)�q�?�(�A�33C���?�\)��Q�B�\�Q�C�e                                    BxuBqF  "          @�?�z��xQ�@   AʸRC���?�z�����aG��+�C�e                                    BxuB�  �          @���@G��u?@  A��C�U�@G��l�Ϳ��\���RC��                                    BxuB��  T          @�33@p��u?�(�AuG�C���@p��|(��^�R�/33C��f                                    BxuB�8  T          @��H?��H�p  ?�  A�\)C���?��H���
��p����\C��R                                    BxuB��  
�          @��
@ ���I��@��A�z�C�Ff@ ���vff>\@�  C��                                    BxuB��  
�          @�=q@
�H�A�@)��B��C��\@
�H�|��?G�A�C�ff                                    BxuB�*  �          @���?�\)�G�@,��B33C�y�?�\)���?G�AffC�ff                                    BxuB��  �          @�z�?�=q�L(�@0��B��C��=?�=q��z�?L��A z�C��=                                    BxuB�v  
�          @�Q�?xQ��:=q@R�\B<33C�C�?xQ����?�z�A���C��
                                    BxuB�  T          @���?.{�9��@Y��BCQ�C��f?.{���R?�G�A��C���                                    BxuC�  T          @���?Y���@  @P��B8��C���?Y����
=?�=qA�
=C��R                                    BxuCh  
�          @�Q�?(���0��@_\)BK��C���?(����(�?�z�A���C���                                    BxuC!  �          @�z�?5�5�@fffBL  C�f?5���?�(�A�(�C��)                                    BxuC/�  "          @��>���0  @l(�BT�HC���>����
=?�A�(�C��q                                    BxuC>Z  �          @�=q>�{�*=q@l��BXQ�C���>�{��z�?��A��
C�aH                                    BxuCM   �          @��>�(��(�@xQ�Be�RC���>�(���G�@
=qA߮C�f                                    BxuC[�  �          @��>�G��
=q@\)Bt=qC��=>�G��vff@��A��C�G�                                    BxuCjL  T          @��R>���(Q�@]p�BQffC�n>����  ?ٙ�A�33C��                                    BxuCx�  T          @�  >����7
=@[�BH=qC�O\>�����?���A�Q�C�C�                                    BxuC��  �          @�{?J=q�?\)@EB3��C�k�?J=q��(�?��HA��\C�n                                    BxuC�>  "          @��?+��Mp�@#33B�C��?+�����?!G�AQ�C��{                                    BxuC��  
�          @���@G��o\)?h��A;�C���@G��l�Ϳ�ff�Z�RC��H                                    BxuC��  �          @�33@!��q�?�@�p�C��=@!��dz`\)����C��                                    BxuC�0  
�          @�
=@'��y��>�=q@QG�C���@'��aG���Q���  C�Q�                                    BxuC��  �          @�Q�?&ff�L��@{Bp�C��H?&ff��  ?z�AG�C��                                     BxuC�|  
�          @��R>���C�
@FffB4�HC�j=>����{?�Q�A}p�C��                                    BxuC�"  
Z          @�\)>B�\�Mp�@E�B/�C��q>B�\����?���Af{C�K�                                    BxuC��  "          @�G����
�q�@\)B�C�⏼��
����>��R@~{C��f                                    BxuDn  "          @��?\)�w�?�33Aϙ�C��?\)�����u�J�HC���                                    BxuD  �          @��H?���j�H?��
A�  C�Ff?����=q�u�S33C�}q                                    BxuD(�  T          @�(�?xQ��vff?�ffA�
=C��?xQ�������
���C�q�                                    BxuD7`  �          @�p�?+���=q?��A�p�C��=?+������&ff�(�C�j=                                    BxuDF  
�          @�z�?0���`  @"�\B�C���?0����G�?   @��HC���                                    BxuDT�  
�          @��R?��
�g
=@��B=qC��\?��
��=q>���@�
=C��                                    BxuDcR  �          @�\)?���XQ�@,��B(�C��?������?5A�RC�'�                                    BxuDq�  
Z          @�G�>��R�P��@HQ�B.��C��q>��R��(�?��AiC��                                    BxuD��  
�          @�G�?�{�U�@%�B�C�,�?�{��p�?!G�A��C�n                                    BxuD�D  
�          @�\)?��H�k�?���A���C�t{?��H��33�L���#33C�N                                    BxuD��  �          @�p�?�\)�A녿8Q��=�C��?�\)��\�ff�33C��=                                    BxuD��  "          @�G�?���E�HQ��#\)C���?�׿�
=��\)�y\)C���                                    BxuD�6  
(          @�p�?�p��l(��ٙ����\C�R?�p�����QG��9�
C���                                    BxuD��  T          @�Q�?}p��b�\@'
=BG�C��
?}p����?\)@��C�k�                                    BxuD؂  
�          @�=q?\(��xQ�@Q�A�  C�AH?\(����#�
�\)C��                                     BxuD�(  �          @���?�\)��z�?��
A��HC�#�?�\)����aG��3�C��\                                    BxuD��  �          @��\?���
=?
=q@ٙ�C���?��|�Ϳ�����ffC�o\                                    BxuEt  �          @��?��R��\)?:�HA\)C��3?��R������33��=qC�
                                    BxuE  �          @��H?����z�?��AV�HC���?����zῈ���W�C��                                    BxuE!�  �          @�(�?�=q���ͿE��
=C�G�?�=q�^�R�333��C��R                                    BxuE0f  T          @��?��n�R��R��G�C�q?�����p  �Rp�C���                                    BxuE?  
�          @���?��H�k��8Q��%C���?��H�H�ÿ�
=����C���                                    BxuEM�  �          @�?��R�u@p�A�\)C��{?��R��=�Q�?���C�G�                                    BxuE\X  "          @�{?������?�p�As�C�&f?�����{�fff�1�C��                                    BxuEj�  �          @�z�@z����H?.{A\)C�s3@z��z=q��{����C���                                    BxuEy�  �          @�
=?�
=���
?��HAn=qC���?�
=��{�h���3
=C�XR                                    BxuE�J  T          @��?�ff��{?�\@�p�C���?�ff�z�H�Ǯ���\C�T{                                    BxuE��  �          @��
@Q�����>��@��C���@Q��q녿\��z�C���                                    BxuE��  �          @��@Q����>���@n{C��3@Q��mp���
=���C��                                    BxuE�<  T          @���@\)���׾��R�w
=C��3@\)�X���(�����C���                                    BxuE��  
�          @��H@z����:�H�\)C�~�@z��Tz��(���=qC�n                                    BxuEш  
�          @�Q�?�ff�|�Ϳ���
=C�<)?�ff�#33�dz��C=qC��                                    BxuE�.  
�          @�\)@
=q�w���=q��G�C���@
=q�)���N�R�-�RC��                                    BxuE��  �          @�
=?�
=��  �8Q��{C�7
?�
=�i�����ޏ\C��                                    BxuE�z  
�          @��\?��
��=q@�AظRC��
?��
���
�#�
�#�
C��)                                    BxuF   T          @���?+��hQ�@Mp�B$
=C�8R?+���
=?���APz�C��                                    BxuF�  "          @�z�?+��:�H@n�RBM�C�xR?+����H?�33A���C�c�                                    BxuF)l  T          @�
=�������@�=qB�(�C�� ����i��@<(�Bp�C�Ф                                    BxuF8  �          @�
=<����@�z�Bu�C�T{<��|(�@%�B��C�0�                                    BxuFF�  "          @��R>��
�!G�@}p�BeG�C���>��
���H@33A�{C�E                                    BxuFU^  �          @�?
=q�1G�@n{BS  C���?
=q��ff?�(�A�G�C���                                    BxuFd  �          @�z�?=p��!�@u�B]�C�4{?=p�����@(�A�ffC�9�                                    BxuFr�  T          @��
?�����@u�Bd�C�o\?���z�H@\)A�ffC���                                    BxuF�P  "          @�Q�=�G��b�\@K�B'�
C�ٚ=�G����
?��A^ffC��f                                    BxuF��  �          @�ff�Ǯ����?�{A���C�c׾Ǯ���������C���                                    BxuF��  
�          @��׿��\��(�>���@}p�C��f���\��  �������C�:�                                    BxuF�B  �          @��R�J=q���H�\���C��\�J=q�i�����p�C��q                                    BxuF��  T          @�
=��������=p��#�C�8R�����Q��%��  C��f                                    BxuFʎ  "          @��������{�z���z�C�:ᾅ��[���R��C��
                                    BxuF�4  �          @�{��z���{?�
=A�Q�C�ᾔz����H�O\)� z�C�+�                                    BxuF��  T          @��ÿ
=�\��@N�RB*��C�"��
=���?��RAq��C�N                                    BxuF��  
�          @�33��
=�dz�@N{B'33C��q��
=���?�
=Ab=qC�ff                                    BxuG&  �          @��H����j=q@H��B"z�C�������ff?���AMC��                                    BxuG�  
�          @�ff��
=�Z=q@]p�B4�C�xR��
=��(�?�p�A��C�e                                    BxuG"r  "          @��þL���J�H@x��BK�C�0��L�����?�(�A��HC��H                                    BxuG1  �          @���>\)�2�\@�Ba33C�h�>\)���@=qA�33C���                                    BxuG?�  T          @�  ������@��RB}�C����������@<(�B(�C�                                    BxuGNd  T          @��׾�����@���B�(�C��R�����~�R@AG�B��C���                                    BxuG]
  �          @�  ���33@���B��
C�)���z=q@C�
B�C�<)                                    BxuGk�  �          @��R>.{��=q@�=qB�Q�C�R>.{�R�\@8Q�B$��C�~�                                    BxuGzV  T          @�?��?�@�Q�B���B/�R?�Ϳ�{@��B�W
C���                                    BxuG��  T          @��?���?
=@���B�p�A߮?��Ϳ���@��B���C��)                                    BxuG��  �          @���?�(�?u@z=qB��B  ?�(��s33@z=qB��C��                                    BxuG�H  �          @�ff��\�ff@j�HBc{C�� ��\�q�@�A�  C�&f                                    BxuG��  �          @�G��:�H��=q@��\B�k�CxaH�:�H�`  @1�Bp�C�#�                                    BxuGÔ  
�          @��\��G��
=@���Bw{C�q��G��n{@&ffB
=C��H                                    BxuG�:  �          @��R�Ǯ��{@���B�\)C��Ǯ�`  @.{BC���                                    BxuG��  
`          @��R�L�Ϳ�(�@\)B  C�B��L���e�@)��B��C��
                                    BxuG�  �          @�
=��(���Q�@uB�33C�\��(��P  @(��B=qC�33                                    BxuG�,  �          @�z᾽p��z�@aG�Bl�
C�����p��\��@(�B{C��                                    BxuH�  �          @�\)�5�\@o\)B��RCu��5�C�
@(Q�B   C�z�                                    BxuHx  �          @�{���?E�@o\)Be��C!���׿��\@k�B`Q�CL8R                                    BxuH*  �          @�  �   ?��@`��BLC@ �   ��(�@n{B_
=C=��                                    BxuH8�  �          @�ff�(�?:�H@p  BiffC!�
�(����@j�HBa�
CM��                                    BxuHGj  �          @�33��?.{@mp�Bm�C!�\������@g
=Bd�CO=q                                    BxuHV  �          @���!�?�G�@g�BP33C��!녿�@s33B_33C?�H                                    BxuHd�  �          @��׿���?5@~�RBy�C��������@w�BnffCR�3                                    BxuHs\  �          @�33�У׽�G�@���B��C7ٚ�У׿�@c33BZp�Ce�R                                    BxuH�  T          @�ff�ٙ���@u�B��C8^��ٙ�����@W
=BU  Ccn                                    BxuH��  �          @��׿�(����
@|��B��C4�3��(���ff@`��Bb(�Cf�3                                    BxuH�N  �          @�G�������
@���B�B�C8aH�����Q�@j�HBk�\Cqc�                                    BxuH��  T          @��\��{��Q�@�(�B��\CE�R��{��@a�B]�Cs)                                    BxuH��  �          @���z�=���@�{B��C*�R�z��ff@r�\B}��C|                                      BxuH�@  T          @��=u>�ff@�Q�B��B�=u�\@�Q�B��\C�!H                                    BxuH��  T          @�33�(�>���@���B��C�׿(���33@|��B�
=Cy�R                                    BxuH�  �          @��H��
=�   @��B�=qCG���
=��@hQ�BR�Cn��                                    BxuH�2  �          @��
��Q쿆ff@�  B�u�CW�3��Q��5@U�B933Cs�                                    BxuI�  �          @��H��33��  @y��B�� C]�ÿ�33�7
=@;�B*Q�Cs޸                                    BxuI~  �          @�
=���
=q@���B�L�CL�������@e�BUz�Cs�\                                    BxuI#$  �          @�����R��@��
B��RCK쿞�R��
@]p�BS�Cq��                                    BxuI1�  �          @�G���=q=L��@���B�p�C2W
��=q��ff@o\)B^33C`��                                    BxuI@p  �          @��
���þ\)@��B��C8c׿���� ��@o\)BX=qCc�f                                    BxuIO  �          @����\)��@��\B���C5.��\)��@i��B`{Cd��                                    BxuI]�  �          @�녿�(��@  @s33B�{CO8R��(��
=@Dz�B?p�Cn(�                                    BxuIlb  �          @��ÿ��>��R@��B�� C#:῅����@u�B�33Cl\                                    BxuI{  �          @�����>�\)@�{B�ffC'𤿨�ÿ���@x��Bw��Cf�
                                    BxuI��  
(          @�=q��z�>�@z�HB��C$uÿ�zῨ��@n{BqQ�CZh�                                    BxuI�T  T          @���?�{@c33BV�C�q��=���@�Q�B��{C0޸                                    BxuI��  �          @�p���G�?�
=@p  Bkp�C�f��G���p�@�  B��{C?�
                                    BxuI��  
�          @����H?��R@E�B?(�C�=���H>�(�@h��Bt�\C'��                                    BxuI�F  T          @�ff��p�@�
@U�BRB��\��p�>Ǯ@x��B�
=C%ff                                    BxuI��  �          @�{�ٙ�@�R@EB?33B���ٙ�?&ff@p��B�aHC{                                    BxuI�  �          @����p�@��@Dz�B>��C 33��p�?!G�@n{B~�RC�                                    BxuI�8  �          @��ÿ�G�@@��?�33A�B�(���G�@ff@%�B?ffB�=                                    BxuI��  �          @|(����R@<(�?�\A�Q�B�����R?�\)@7�BP=qB���                                    BxuJ�  T          @�����@�R@E�BBp�B�\)���?(��@o\)B���C�R                                    BxuJ*  �          @����=q@�H@G
=B6�B�G���=q?Q�@vffBz  C޸                                    BxuJ*�  "          @��\��Q�@,��@[�B7z�B�p���Q�?p��@�Q�B|��C.                                    BxuJ9v  T          @�z�Tz�@<��@%�B �B�z�Tz�?��@fffB�33B���                                    BxuJH  �          @�{>k�@g
=@�
A�\)B��
>k�@
=@Z�HB]G�B�\                                    BxuJV�  T          @�{���@0  @<��B3{B��f���?���@vffB��{Cu�                                    BxuJeh  "          @��� ��@�\@y��BV��C�f� ��>��@�z�B���C/�                                    BxuJt  "          @��\���@�@��HBW=qC	B����=u@��B��{C2T{                                    BxuJ��  T          @�{��?�z�@w�BO33C(���=L��@�=qBtC2�R                                    BxuJ�Z  T          @�=q���@��@��HBY�C�׿��>�\)@�p�B��qC+z�                                    BxuJ�   
(          @�녾Ǯ@8��@��HBZ  B�8R�Ǯ?Y��@�{B��{B��)                                    BxuJ��  �          @��R��G�@5@~{BG��B󞸿�G�?\(�@���B���C                                      BxuJ�L  
�          @�Q��ff@�\@��BW(�C�{�ff>�  @��B��=C-W
                                    BxuJ��  �          @�ff�33?ٙ�@���Ba33CxR�33����@�33B}�C;k�                                    BxuJژ  �          @�{�Q�?�{@��\B`z�C8R�Q��@���BqG�C@=q                                    BxuJ�>  "          @���'�?�
=@�=qB[(�C���'��+�@�{Bd��CB\)                                    BxuJ��  �          @�\)�J�H?   @s�
BG��C*�R�J�H��@j=qB=\)CHO\                                    BxuK�  �          @�\)�Tz�=���@l��B@�\C2W
�Tz῾�R@X��B,  CL33                                    BxuK0  �          @���E��.{@y��BN�\C7)�E���@]p�B033CRp�                                    BxuK#�  �          @��
�%�L��@�33Bm  C5��%���@z=qBJCX33                                    BxuK2|  �          @��R����?��@�=qBu��C� �������@�G�B�.CC��                                    BxuKA"  �          @��R� ��?�G�@�=qBt��C(�� �׿�@���B��CC                                    BxuKO�  T          @�(���p�?�@�Q�Bv�CQ��p���R@�{B�G�CEff                                    BxuK^n  T          @�(��޸R?���@���Bz{C	xR�޸R���@���B��qCC!H                                    BxuKm  "          @�33�
=q?�(�@�33Bk��C޸�
=q�   @��B�\CA)                                    BxuK{�  
�          @����#�
?��
@��
B]  C���#�
�z�@���Bi��C@                                    BxuK�`  �          @��H�$z�?�(�@��B^z�C�=�$z�#�
@�G�Bi33CA�R                                    BxuK�  �          @�녿�{@C�
@c33B3=qB�{��{?��
@�Q�B}��Cc�                                    BxuK��  "          @�G����@�R@tz�BEffC�����?&ff@�Q�B~z�C"�R                                    BxuK�R  
�          @����
@��@j=qB=�\C@ ��
?0��@�33Bt{C#:�                                    BxuK��  
Z          @�(��Q�@4z�@aG�B.�RC.�Q�?��@�(�Bl�\Cp�                                    BxuKӞ  
�          @�(��"�\?�{@��BQ  CǮ�"�\�#�
@��RBq��C4&f                                    BxuK�D  �          @���(�?�@��B]C��(��u@���B��C5}q                                    BxuK��  
�          @�  ��p�?˅@���Bm��C.��p���33@�=qB���C=�                                    BxuK��  �          @���?�ff@�G�BhG�C�
����@�{Bv{CB\)                                    BxuL6  "          @���Q�?���@��RBqz�C&f�Q���@�(�B��HCC��                                    BxuL�  "          @���Q�@ ��@��
B`��C���Q�<��
@��B�(�C3��                                    BxuL+�  "          @�33�G�?�Q�@��Bi�C=q�G���@�ffB��=C7��                                    BxuL:(  T          @�\)��G�?���@�p�B|C
8R��G��   @���B��CC                                    BxuLH�  �          @�����H?˅@��Br�RCٚ���H����@�  B�aHC?��                                    BxuLWt  �          @�  ��\?�\)@�G�Bv\)C+���\��R@�ffB�.CD�q                                    BxuLf  
�          @�(���>��@��\B��qC&E����{@�=qB|33C_�                                    BxuLt�  
�          @���33?�@�  B���C#8R��33��
=@�=qBz
=CX�                                    BxuL�f  "          @������=L��@���B��C2{�������H@��BuCixR                                    BxuL�  "          @��R��G�>��
@�  B��C%�ῡG���G�@�{B���Cjff                                    BxuL��  �          @�{��p��@  @�
=B��qCH�3��p��'
=@�  BJ�
Chٚ                                    BxuL�X  "          @��
�!G�?�z�@��
BW  C��!G��#�
@�{Bq�C7��                                    BxuL��  �          @�z��1G�?�@z�HBGp�C޸�1G�=L��@��BeG�C2��                                    BxuL̤  "          @�ff�(��?�p�@��
BR33C��(�ý�Q�@�
=BmC5�q                                    BxuL�J  T          @�{�!�?�  @�Q�B]\)CL��!녾�Q�@�Q�Bq��C;�q                                    BxuL��  �          @�ff�!�?���@�z�Bg=qCz��!녿E�@��RBm33CD�                                    BxuL��  T          @�\)��?���@�p�Bg33C�f�����@��HBu�C@�                                    BxuM<  
�          @����
?�{@�ffBp=qC5���
�G�@���BvCF��                                    BxuM�  �          @����\)?�  @�BgffCW
�\)�#�
@��BrQ�CB�                                    BxuM$�  �          @�=q�!�?B�\@��Bp��C#O\�!녿�@���Bi�RCL                                    BxuM3.  "          @�(��\)>�(�@�  Bw�HC*5��\)�\@���Bd�CSQ�                                    BxuMA�  �          @�(��>8Q�@�=qB��C/�{���\@��Bc(�CY
=                                    BxuMPz  �          @��
��\<�@���B�C3Q���\��33@��B_��C[��                                    BxuM_   
�          @���p�>�z�@���B��C,� �p���33@���Bj�CX                                    BxuMm�  �          @�(���R>\)@�33B���C0p���R��@�  Be��C[&f                                    BxuM|l  T          @����	��=���@�ffB�\C1aH�	�����@��\Bh(�C]G�                                    BxuM�  �          @��Ϳ�33=���@�G�B��{C0��33��@�p�BoCa.                                    BxuM��  �          @�z��\)=�@�(�B��C/����\)��
=@�Q�Bxz�Cf�                                    BxuM�^  �          @��H��>k�@�Q�B�
=C,������\@�{Bv��C`z�                                    BxuM�  
�          @�\)�   ?�ff@�\)Bk�
C
�   ��@��\B��
C7p�                                    BxuMŪ  T          @�{��?���@��Bj  C�\����\)@�B��C;O\                                    BxuM�P  "          @�Q���?8Q�@��\Bt=qC#� ����33@��Blz�CMQ�                                    BxuM��  "          @��R�{=L��@�ffB�p�C2���{��@��\Bb\)C[+�                                    BxuM�  T          @�\)��
>�=q@���B�  C,�f��
���@��Bn��CZxR                                    BxuN B  T          @���+�=L��@��\Bn��C2��+���  @�\)BSz�CU8R                                    BxuN�  
�          @�G����B�\@�33B�G�C9(�����@�(�B`�\C`޸                                    BxuN�  �          @�����R��(�@���BU��CZ� ��R�_\)@J=qB�RCj��                                    BxuN,4  	�          @�33�녿��@�33BjG�CL�����#�
@W
=B1�
CdY�                                    BxuN:�  "          @�  �$z�>�{@��Bj33C,�=�$zΎ�@���BY{CO�H                                    BxuNI�  
�          @�
=�=q>�@|(�BhQ�C)!H�=q��{@s33B\�\CL��                                    BxuNX&  "          @�=q��33>�z�@���B���C(E��33��ff@tz�B}�
C^�                                    BxuNf�  �          @�p��Y���$z�@���Bi��C{���Y�����\@C33B{C�)                                    BxuNur  T          @�ff�\(��*=q@�(�Be��C|�\(�����@@  B(�C�#�                                    BxuN�  �          @��׿Tz��&ff@�z�Bh\)C|LͿTz���33@B�\B��C�E                                    BxuN��  
�          @�G��h���<��@��BU��C|�=�h�����\@,��A���C�
=                                    BxuN�d  T          @�녿(���,(�@�Q�Bi��C�*=�(����\)@G�B��C��R                                    BxuN�
  
�          @�����33�{�@`  B&��C�t{��33��
=?�(�A�ffC��q                                    BxuN��  N          @�
=�������@<��B
ffC�E������\?�=qA@(�C���                                    BxuN�V  
�          @�\)������@�A��C�������
=>B�\@Q�C���                                    BxuN��  "          @�
=>.{��
=@5�A��C���>.{��p�?5@�C�ٚ                                    BxuN�  �          @�>B�\��@G�B�\C�>�>B�\����?�Q�AJffC�
=                                    BxuN�H  T          @��?z��N{@�Q�BK33C�
?z���  @{A��C���                                    BxuO�  "          @��ýL���o\)@mp�B3�C�� �L�����
@   A��HC���                                    BxuO�  
�          @�Q�<��
��@P��BffC�#�<��
���H?�=qA_\)C�q                                    BxuO%:  �          @��R�#�
���\@>�RB��C�Ǯ�#�
���
?�G�A*ffC��\                                    BxuO3�  "          @�33>��.�R@z=qBZ�C���>���Q�@%�BG�C�=q                                    BxuOB�  
Z          @���<��
�g
=@'�BC�1�<��
���H?��AZ�HC�*=                                    BxuOQ,  �          @��ͿJ=q��?�A��C��
�J=q��=q=���?�p�C�^�                                    BxuO_�  
�          @����z�����@*=qB
=C��3��z���Q�?p��A1�C�AH                                    BxuOnx  
�          @��
��33�U@Z=qB5�C�f��33��(�?�33A���C���                                    BxuO}  T          @��H@�H�:=q?�G�A��C��3@�H�QG�>���@�  C�:�                                    BxuO��  T          @�@G��\)@XQ�Bh��C�4{@G����@:�HB<�HC�9�                                    BxuO�j  
�          @�{?�  ?�G�@~�RBG�B 
=?�  ����@��B��=C��)                                    BxuO�  �          @�?�G�?�p�@�  B��B1ff?�G���Q�@�p�B�.C��                                    BxuO��  �          @���?�
=?\)@���B�z�A�z�?�
=����@�p�B��C���                                    BxuO�\  T          @�G�?�Q�?L��@��RB�Q�A�\?�Q�Tz�@��RB��C��                                    BxuO�  �          @�ff?��?k�@xQ�B��=BG�?���
=@|(�B�u�C���                                    BxuO�  
�          @�\)?u�aG�@�A�G�C��f?u��G�?�RA  C��{                                    BxuO�N  "          @��\?����=q?z�HAF{C��q?�����
�333���C���                                    BxuP �  �          @�G�?���p�?O\)A%�C�W
?����ͿaG��4z�C�Y�                                    BxuP�  T          @�33?�\)�(�@�=qB�#�C�˅?�\)�z�@c33Ba��C�@                                     BxuP@  	�          @�33?��;�@�  B�ǮC���?��Ϳ�33@b�\B[�
C�f                                    BxuP,�  
�          @�Q�?�p�?�ff@dz�Bgp�A�p�?�p���{@mp�Bv{C�&f                                    BxuP;�  
�          @�\)?}p��h��@�Q�B�
=C���?}p��=q@g�BYG�C�&f                                    BxuPJ2  �          @�G�?���s33@{�B���C�k�?����@S�
BP��C��f                                    BxuPX�  !          @�p�@33?��@_\)BV�HB�H@33>��@s�
Bv��@�=q                                    BxuPg~  "          @��@0  ?�33@;�B)ffA��
@0  >��@U�BH
=A=q                                    BxuPv$  T          @�33@ ��?�\)@L(�B@��A���@ ��=�G�@^{BXp�@{                                    BxuP��  "          @���@�\?��@\(�BVffA�p�@�\�u@g
=Be�\C��                                    BxuP�p  �          @�z�@%�����@^{BT�C��3@%����@G�B8\)C���                                    BxuP�  
�          @��@G����
@j=qBt�C��@G���33@XQ�BX{C��q                                    BxuP��  T          @��
?�33?�ff@vffB}z�BU�
?�33=L��@���B�@�R                                    BxuP�b  T          @��H?��\>B�\@��\B�z�A'�?��\��ff@w�B�C��                                    BxuP�  �          @�(�@���L��@`��BX�C���@��� ��@?\)B.z�C��                                    BxuPܮ  
Z          @��H@   ��
=@N{B@p�C�q@   �   @�RBQ�C���                                    BxuP�T  T          @���?���(�@K�B>��C�g�?���K�@�A��
C�T{                                    BxuP��  T          @�33@%����@*=qB�C��@%��@��?ٙ�A�
=C�L�                                    BxuQ�  
�          @��
@�
�/\)@-p�B\)C�t{@�
�a�?�ffA��C��                                    BxuQF  
�          @���@녿�(�@8Q�B.�RC���@��7�?�p�A��C�8R                                    BxuQ%�  T          @���@{��z�@e�B\�C��@{��ff@O\)B?�HC���                                    BxuQ4�  "          @�=q@1�>�33@e�BO�@�R@1녿s33@^{BF�HC��
                                    BxuQC8  �          @��H@AG����\@H��B/{C��q@AG��33@�RB��C�XR                                    BxuQQ�  �          @�ff@;���@B�\B!�\C���@;��>�R@�A���C�K�                                    BxuQ`�  T          @���@C�
�0��@ffA�\)C���@C�
�Z=q?�(�AmC��                                    BxuQo*  �          @�p�@*=q�c�
@�
A���C�j=@*=q���?(��@���C��f                                    BxuQ}�  
�          @���@$z��K�@�A��HC�y�@$z��s�
?��AV{C���                                    BxuQ�v  T          @�(�@,(��У�@VffB;33C�c�@,(��-p�@#�
B\)C�g�                                    BxuQ�  
#          @�Q�@8���HQ�@%�A�\)C�` @8���u�?��At  C��H                                    BxuQ��  
�          @�
=@,���~�R?�(�A��
C��@,����p�>�G�@��
C��
                                    BxuQ�h  
�          @�z�@<����=q?�z�A�(�C��)@<����\)>�Q�@s�
C���                                    BxuQ�  
Z          @�p�@Mp��\)?�
=A�33C�l�@Mp����\>8Q�?�{C�N                                    BxuQմ  
�          @�(�@\���_\)@
=qA�  C�XR@\������?L��A	C�O\                                    BxuQ�Z  
(          @�\)@x���B�\@��A�z�C��@x���k�?�
=AFffC�L�                                    BxuQ�   �          @�G�@B�\�Q�@0  A�z�C�s3@B�\����?�
=A}C��=                                    BxuR�  �          @���@o\)�%�@!G�A�  C��)@o\)�R�\?�(�A�33C�W
                                    BxuRL  �          @���@w
=�p�@{A�{C��@w
=�6ff?��Au�C��\                                    BxuR�  
�          @�=q@g��   @   A�C��3@g��A�?�G�AA��C�H                                    BxuR-�  
�          @�=q@�(��33@��A֣�C�w
@�(��>�R?���A~{C�
                                    BxuR<>  �          @�z�@�(���@p�AٮC���@�(��HQ�?�(�A~{C�j=                                    BxuRJ�  
�          @��@�p��3�
?��RA��HC��@�p��U�?fffAp�C��
                                    BxuRY�  
�          @��R@���*=q?�A��\C�q�@���G�?G�A�C�]q                                    BxuRh0  �          @�Q�@���333?�(�A�C�Ǯ@���N{?&ff@ڏ\C��                                    BxuRv�  �          @���@���G�?ٙ�A��C�Ф@���`��?�@�
=C�*=                                    BxuR�|  
�          @�=q@���N{?�Al��C���@���`��>u@�HC�q�                                    BxuR�"  
�          @�G�@���L��?���A<��C��H@���X�ü�����C��                                    BxuR��  T          @�  @���H��?z�HA$  C�H�@���Q녾����ffC��=                                    BxuR�n  �          @���@����A�?h��A�C�!H@����J=q�.{��ffC��3                                    BxuR�  �          @���@����C33?Q�A	�C�f@����I������,��C���                                    BxuRκ  
(          @��@��H�C�
?!G�@ϮC�*=@��H�E��ff��33C��                                    BxuR�`  �          @��
@����<��>aG�@  C�!H@����6ff�J=q���C���                                    BxuR�  T          @�(�@�33�6ff���Ϳ�ffC��@�33�)������-C���                                    BxuR��  
�          @��@�(��,(����H��\)C��@�(�����{�a�C�
=                                    BxuS	R  �          @�=q@��
�B�\�z���Q�C�S3@��
�*�H������G�C�                                      BxuS�  �          @�G�@�p��:�H�
=����C���@�p��#33�����
=C���                                    BxuS&�  �          @���@���:�H�:�H���C���@���   ��z���p�C��q                                    BxuS5D  T          @�Q�@�z��1G����
�,  C��{@�z��G���33���C���                                    BxuSC�  "          @�Q�@���/\)����-C�@���\)������HC�*=                                    BxuSR�  �          @��
@�{�5�n{� (�C��@�{��������p�C��3                                    BxuSa6  �          @��R@��H�'
=�����`��C�,�@��H�G��
=����C��                                    BxuSo�  "          @�
=@���N{�(���C�� @���5��33���\C�t{                                    BxuS~�  �          @�ff@��\�0  �z�H�&=qC�}q@��\�G�����{C�˅                                    BxuS�(  T          @��@�{�:=q�Y�����C�]q@�{�{��\��G�C�w
                                    BxuS��  
�          @��@����?\)�^�R��\C��q@����!녿�ff��
=C��H                                    BxuS�t  "          @���@��W
=�^�R�
=C�l�@��8Q��
=����C�z�                                    BxuS�  T          @��@���k�����)��C��=@���X�ÿ��H�l��C��                                    BxuS��  �          @���@��
�s33���
�8Q�C�}q@��
�e���\�IC�Q�                                    BxuS�f  T          @�
=@����p��>���@���C�aH@����k��Y�����C��=                                    BxuS�  �          @���@���_\)�����733C�  @���<(�����p�C�`                                     BxuS�  T          @���@���]p����H�D��C��H@���8Q��G����C�%                                    BxuTX  �          @��@�z��^{�Y���
�HC���@�z��@  ��Q���(�C�Ф                                    BxuT�  �          @���@�p��]p�������C��R@�p��E��z���
=C���                                    BxuT�  "          @�{@g
=�<(���\��\)C�j=@g
=���8���
G�C��=                                    BxuT.J  
�          @��@B�\�(Q��\)��z�C��=@B�\��
=�Mp��+
=C��                                     BxuT<�  
�          @�=q@H���'��@  �C�
@H�ÿ�G��k��:�C�'�                                    BxuTK�  �          @�
=@
=q�����g��O\)C���@
=q�z������s��C�w
                                    BxuTZ<  T          @���?˅�������\�C��{?˅>L�������@��
                                    BxuTh�  
Z          @���?�=q��z����HaHC��?�=q>�����\)p�A+�                                    BxuTw�  T          @�33@�\��  �s33�aQ�C��f@�\�L�����\�}(�C�=q                                    BxuT�.  T          @�(�@J=q���H������\C�!H@J=q�����1G��{C�.                                    BxuT��  
�          @�(�@_\)�\)�G���C��q@_\)��p��(Q���C�}q                                    BxuT�z  
�          @��@\(��7
=�Ǯ���C�(�@\(��p������{C��                                    BxuT�   �          @���@u���H���H�o�C���@u���z����\)C��H                                    BxuT��  �          @��H@y���G��޸R��=qC�޸@y����{����\C��\                                    BxuT�l  �          @�Q�@u�����G���(�C���@u����� ��� Q�C��3                                    BxuT�  
�          @��@`�׿���33��p�C��@`�׿���1G��\)C��
                                    BxuT�  T          @��H@A��'��{���C��3@A녿��
�<(���C��                                    BxuT�^  "          @�G�@�R�H�������C�!H@�R�z��>�R�$�C�n                                    BxuU
  
�          @��H@(��S�
�   ��C�+�@(�� ���<(�� =qC�                                    BxuU�  �          @��@\)�0���1G���RC��@\)��  �`  �C�
C�n                                    BxuU'P  
�          @�{?��H�\�������
C���?��H�!G��X���;  C��R                                    BxuU5�  
(          @�\)?�z��#�
�\(��<�RC�g�?�zῬ����=q�pp�C�g�                                    BxuUD�  T          @�{?�ff�	���|���f{C��?�ff�O\)���)C��
                                    BxuUSB  	�          @�33@��
�H�^�R�D�RC��
@��z�H�~�R�p33C�\)                                    BxuUa�  �          @���@��p��X���;��C��@녿���z=q�e�\C��{                                    BxuUp�  �          @�
=@>�R�*=q��
��Q�C�'�@>�R����1��{C�ٚ                                    BxuU4  �          @�p�@h���G������H��C��R@h���(Q��p���Q�C��                                    BxuU��  T          @�ff@\���QG������x��C�@ @\���,���G����C��{                                    BxuU��  �          @��@J=q�Vff��������C���@J=q�/\)�=q��C���                                    BxuU�&  "          @�G�@G���=q=��
?s33C��{@G��x�ÿ�
=�j�HC��                                    BxuU��  "          @��@=q��zᾳ33��{C�{@=q�u���{��z�C��                                    BxuU�r  �          @���?����{�k��3�
C�c�?���|(��\)��=qC�aH                                    BxuU�  
�          @��H@���vff��\)��=qC�� @���J�H�.{��RC�b�                                    BxuU�  "          @��@E��\�Ϳ������C��@E��3�
�#33��C��R                                    BxuU�d  �          @��@Tz��N�R������Q�C��@Tz��*=q�33��
=C��=                                    BxuV
  
Z          @��@Q��E��\��{C�Y�@Q�����(Q���RC���                                    BxuV�  �          @�\)@Vff�C33�z��ĸRC��q@Vff�G��9���z�C��\                                    BxuV V  
�          @���@]p��/\)�ff�˙�C���@]p���(��5���RC�#�                                    BxuV.�  O          @��@]p��'��	���ҸRC�t{@]p����5��C��)                                    BxuV=�  
-          @�{@X�ÿ\(�����  C���@X�þ���
=�ffC��R                                    BxuVLH  "          @��@c33��{����C�:�@c33>�����H�Q�@�                                      BxuVZ�  T          @�ff@Z=q��Q��#33��
C�j=@Z=q��Q��2�\�ffC��R                                    BxuVi�  �          @�
=@Tz��G�����Q�C�f@Tz�u�4z���C��{                                    BxuVx:  �          @��@\(����
=q��C�p�@\(���
=�(�����C���                                    BxuV��  �          @�z�@w
=��G��ff��(�C��\@w
=����"�\�ffC��                                    BxuV��  
�          @��
@dz�\)�:=q���C�� @dz�?=p��4z���
A:�\                                    BxuV�,  �          @�
=@\(�>��R�4z��
=@��@\(�?���&ff�A�=q                                    BxuV��  �          @�z�@_\)�B�\�(���33C�h�@_\)?���%���AQ�                                    BxuV�x  �          @�
=@\��>�
=�333�z�@߮@\��?��R�#33�G�A��
                                    BxuV�  �          @�ff@[���G��4z��ffC��@[�?:�H�.{�=qA?\)                                    BxuV��  T          @�
=@i��>����"�\�
�H@�\)@i��?��������A~�R                                    BxuV�j  "          @�z�@W�?��
�@���"=qA�@W�?�\)�$z��  A�R                                    BxuV�  "          @�
=@���=q�\(��`z�C���@���33���
��
=C��                                    BxuW
�  
�          @���?�=q���H��\)�c�
C��?�=q�tzῼ(���33C��                                    BxuW\  T          @��R?�33��{�   �θRC�C�?�33�w
=���H��
=C�                                      BxuW(  �          @��?�G����=�\)?W
=C��H?�G���{��(��|��C�                                    BxuW6�  "          @�G�?�
=��33����(�C��{?�
=��Q�޸R��ffC�.                                    BxuWEN  T          @�=q?��R���\�����z�C�f?��R�p�׿������C���                                    BxuWS�  �          @�=q@\)�n�R�����c\)C�u�@\)�P  �����Q�C�E                                    BxuWb�  
�          @�\)@!G��O\)��(���C��
@!G��%�%�=qC�
                                    BxuWq@  T          @���@(��8���Q��=qC���@(��z��G��8  C�G�                                    BxuW�  �          @�  @
=�>{�!G��
C���@
=�
=�Q��>�C��                                    BxuW��  "          @���@=q�E��z����C��q@=q�ff�7��"
=C��
                                    BxuW�2  
�          @��@�\�\(�������C�N@�\�/\)�5����C�N                                    BxuW��  
�          @�
=@��^�R��������C�q@��3�
�0  ��HC��\                                    BxuW�~  
�          @�@���Z�H��G����HC�@���0���*�H���C��)                                    BxuW�$  
�          @���@�R�HQ�����  C�1�@�R����8����
C��R                                    BxuW��  T          @��@0  �QG�������Q�C�
=@0  �*�H�{�=qC��                                    BxuW�p  O          @��@>�R�N{���\����C�]q@>�R�.{�����G�C���                                    BxuW�  
_          @��\@R�\�C33�^�R�/33C���@R�\�*�H�ٙ���33C�w
                                    BxuX�  
Z          @��
@C�
�N�R��{�`Q�C�� @C�
�1녿��R��\)C��                                    BxuXb  T          @�{@A��P  ����{C�~�@A��-p���\��33C��                                    BxuX!  
�          @��@Mp��E��{���C��@Mp��%�����p�C���                                    BxuX/�  
�          @�@L(��J=q��ff��p�C��f@L(��*=q�	���ڸRC��                                    BxuX>T  �          @�@S�
�I�����
�J�RC�/\@S�
�.�R�����G�C�9�                                    BxuXL�  T          @�ff@Vff�L(��Y���&�\C�8R@Vff�4z��(���\)C���                                    BxuX[�  
�          @��R@Z=q�J�H�
=���C��f@Z=q�8Q쿺�H����C���                                    BxuXjF  �          @���@P���P  �����HC��@P���>{��
=��p�C�ٚ                                    BxuXx�  �          @�G�@\���QG��   ��ffC�E@\���@  ��33���C��H                                    BxuX��  "          @��@Tz��Tz�����HC��H@Tz��C�
�������C��\                                    BxuX�8  �          @�@�R�R�\����33C�|)@�R�"�\�@���!\)C��                                    BxuX��  T          @�@Q��Vff�{��G�C�8R@Q��!G��S�
�4\)C�{                                    BxuX��  
�          @���@�\)���>�G�@�  C���@�\)����\)�n{C�C�                                    BxuX�*  
�          @��@�z��Q�L�Ϳ\)C�c�@�z��녿#�
��(�C���                                    BxuX��  �          @��@�Q�������=qC��)@�Q��p��L���C���                                    BxuX�v  
�          @��H@>�R�:�H�����
=C�Ǯ@>�R���(Q��{C�H�                                    BxuX�  �          @�p�@fff�.{��{�\��C�u�@fff��
��������C��                                    BxuX��  
�          @��H@e��*�H�����W�C��f@e��G���\��33C��3                                    BxuYh  
�          @�=q@e�'���ff�U�C��@e�{��p����RC��                                    BxuY  �          @���@\(��)�������\)C�0�@\(����G���G�C���                                    BxuY(�  
_          @�33@fff�녿�Q���z�C��3@fff��p��  ����C�/\                                    BxuY7Z  
�          @��R@e��G��G���33C�J=@e������/\)���C��                                    BxuYF   
�          @���@tz����ff��ffC�7
@tz῎{�(���(�C��                                    BxuYT�  
�          @��H@�����R���H�tQ�C�#�@�����Ϳ�����=qC��3                                    BxuYcL  "          @��
@n�R��
�ٙ���  C��f@n�R�\�����G�C��                                    BxuYq�  
�          @��H@Z�H�����.{��C�5�@Z�H�\)�>�R�"C�\)                                    BxuY��  
Y          @�=q@W����R�,����C��@W��333�@���$�RC�#�                                    BxuY�>  T          @�@U��  ������C���@U���  �<����HC��)                                    BxuY��  T          @�p�@J�H�Q���H���HC��@J�H����>�R�=qC�`                                     BxuY��  "          @��@XQ��
=��p���\)C���@XQ��p��#33�p�C�l�                                    BxuY�0  
Z          @��\@`����\��=q����C�g�@`�׿��H������C��q                                    BxuY��  
Z          @��@n{��ff��p���
=C��@n{���������=qC��                                    BxuY�|  
�          @�@O\)�N�R�^�R�,z�C���@O\)�7����H��ffC�:�                                    BxuY�"  �          @�p�@Vff�6ff���
����C���@Vff���\)��ffC��3                                    BxuY��  
�          @�{@/\)�^{�������RC�'�@/\)�:�H������C���                                    BxuZn  
�          @�{@9���J�H��\��33C�1�@9���%��#�
���C�'�                                    BxuZ  �          @�{@��*=q�P  �0z�C��=@녿�Q��u�]�HC�{                                    BxuZ!�  T          @�ff?�p��"�\�Z=q�:�C���?�p����
�~{�f��C�*=                                    BxuZ0`  �          @�\)@���.{�L���+{C�
@�ÿ�G��s�
�W�
C�<)                                    BxuZ?  P          @�ff@\)�J=q�(Q����C���@\)��XQ��8�
C��q                                    BxuZM�  &          @���?�z��P���0����RC�(�?�z��=q�a��D  C�33                                    BxuZ\R  "          @�z�@�\�Vff�{� C��@�\�$z��QG��3��C�8R                                    BxuZj�  �          @�33?���Tz��#�
�ffC���?���!G��Vff�;G�C�W
                                    BxuZy�  &          @��H?޸R�Vff�&ff�
{C��H?޸R�"�\�X���?  C�=q                                    BxuZ�D  
�          @���?�p��Vff�   ���C�AH?�p��$z��R�\�5�RC�                                    BxuZ��  "          @��R@��>{�>{��C�@����i���K��C���                                    BxuZ��  
Z          @���?�\)�>�R�@���"(�C��?�\)���l(��SQ�C�                                      BxuZ�6  "          @�(�?�z��=p��@  �!�HC�n?�z���
�k��Rp�C�h�                                    BxuZ��  �          @�{?���2�\�O\)�/\)C��?�녿��w
=�^��C��3                                    BxuZт  
�          @���@��Ǯ�u�V�C�XR@��   ���
�n  C��                                    BxuZ�(  "          @�G�@	����{���
�n(�C�T{@	��<#�
�����|�>B�\                                    BxuZ��  
Z          @���@(���33�\)�c�\C��q@(�������
=�y(�C��                                    BxuZ�t  �          @��\@ff�G��J�H�0��C���@ff��{�j=q�U�\C��                                    Bxu[  T          @�
=@0  �!����Q�C���@0  ��ff�@���)C�b�                                    Bxu[�  T          @�\)@9���{�ff����C���@9����\�:=q�"�C�J=                                    Bxu[)f  
�          @�\)@:=q�  �!��	�C�"�@:=q�\�A��*�RC�=q                                    Bxu[8  
�          @�=q@-p��)���!����C��@-p���z��HQ��-�\C�t{                                    Bxu[F�  �          @�=q@&ff�/\)�#�
�p�C��R@&ff���R�K��0�C�N                                    Bxu[UX  �          @�Q�@���<(��"�\�
\)C�h�@���(��Mp��7�C��q                                    Bxu[c�  T          @�  @6ff�0�������C��q@6ff����+����C��{                                    Bxu[r�  �          @�(�@L�Ϳ���N{�2C���@L��>�  �QG��6  @���                                    Bxu[�J  �          @�z�@G��.{�U�9  C���@G�>L���Y���=��@j�H                                    Bxu[��  �          @�=q@U����C�
�)�C��@U�>�{�E��*��@��                                    Bxu[��  �          @���@I���k��G��2�RC��3@I��?z��Dz��/\)A(Q�                                    Bxu[�<  
�          @��\@'
=��33�`���K�C��@'
=�W
=�l(��Zz�C���                                    Bxu[��  
�          @���@<�Ϳ�ff�[��>��C�7
@<�;��e�J33C��{                                    Bxu[ʈ  T          @�33@AG������Q��5�
C��H@AG���  �]p��C  C��q                                    Bxu[�.  
�          @�33@<�ͿG��X���@  C���@<��=�G��^�R�F�R@                                    Bxu[��  
(          @�  @녿�\�Z=q�G�C�\@녿W
=�p  �d\)C���                                    Bxu[�z  "          @�{@�\�,���7
=� �
C���@�\����\���L�HC���                                    Bxu\   T          @���@�z��E��,p�C�� @���H�dz��QffC��                                    Bxu\�  �          @���@�
�\)�S33�=Q�C�Ff@�
��=q�qG��c�RC��3                                    Bxu\"l  �          @�
=@   ��33�o\)�c=qC�w
@   ����~{�{�C�4{                                    Bxu\1  	�          @��@{��Q��l(��^��C��@{�aG��w��p�C�7
                                    Bxu\?�  T          @�
=@���ff�g
=�Y(�C���@��z��x���s��C�'�                                    Bxu\N^  "          @�{?�33�aG��z=q�wG�C��{?�33>\)��  �@��
                                    Bxu\]  
(          @�33?������p���q=qC���?���G��z=q�C�B�                                    Bxu\k�  
�          @�Q�@=q���p  �b�RC��{@=q>�G��qG��c��A#�
                                    Bxu\zP  
�          @�p�@33�����w��`z�C�B�@33�B�\��G��q  C��\                                    Bxu\��  
�          @���@�R��
=�j=q�P
=C���@�R����z=q�d�C���                                    Bxu\��  �          @��@�ÿǮ�g
=�N��C�o\@�ÿ���xQ��fffC��                                    Bxu\�B  �          @��\@!G����[��BG�C�7
@!G��B�\�o\)�[\)C���                                    Bxu\��  �          @��@/\)��\)�QG��2z�C�Ф@/\)�}p��h���M=qC�f                                    Bxu\Î  
�          @���@1��*�H�0  �(�C�@1녿�33�U��2�C���                                    Bxu\�4  T          @�{@{�p  �33��  C���@{�G��:�H�  C�*=                                    Bxu\��  T          @�p�@��p��������C�R@��G��<���{C���                                    Bxu\�  
�          @�  @*=q�e��\)���C�Q�@*=q�:=q�C�
�(�C�9�                                    Bxu\�&  
�          @��@!��g
=��R��
=C�z�@!��8Q��S33�#  C��q                                    Bxu]�  	�          @�(�@(���e�#33���C�*=@(���5�W
=�#��C�p�                                    Bxu]r  "          @�33@���l���#�
��C�l�@���<���Y���'p�C��H                                    Bxu]*  �          @�
=@\)�P  �/\)�{C���@\)�{�]p��2\)C��q                                    Bxu]8�  	�          @���@2�\�N�R�*�H� 33C�k�@2�\�{�X���)�C�B�                                    Bxu]Gd  �          @��\@(���`���"�\��z�C�w
@(���1G��U��${C���                                    Bxu]V
  "          @���@(��QG��=p���C�\)@(��(��k��;\)C��H                                    Bxu]d�  "          @��@%�G��:=q���C�޸@%��
�e�7�
C�,�                                    Bxu]sV  "          @��
@>�R�Z�H�ff��p�C��@>�R�/\)�G���C���                                    Bxu]��  �          @�(�@J�H�Z=q�Q��ď\C�t{@J�H�1��9���
�C�]q                                    Bxu]��  "          @��@E�XQ��
=��(�C�8R@E�,���G����C�q�                                    Bxu]�H  T          @���@7��7
=�AG���C��@7����g��7z�C�Y�                                    Bxu]��  �          @��@���"�\�a��4(�C�@�Ϳ˅�����X�\C�}q                                    Bxu]��  "          @���@+��5��*�H�	��C��
@+���Q��/��C��                                    Bxu]�:  T          @�ff@-p��QG��G��˙�C�Ф@-p��+��0����C��=                                    Bxu]��  T          @�33@0���u��u�8(�C���@0���^�R��\)��Q�C�<)                                    Bxu]�  �          @�\)@.�R�qG��G��z�C��@.�R�]p���
=��z�C�R                                    Bxu]�,  �          @��@.�R�l�Ϳ.{��HC�1�@.�R�Z�H�Ǯ��
=C�E                                    Bxu^�  T          @��
@.�R�j�H�(���C�P�@.�R�Z=q��p����C�S3                                    Bxu^x  
�          @��
@��u?0��A
�HC�0�@��x�þ���N�RC��                                    Bxu^#  
�          @��@���xQ�?c�
A0Q�C��R@���~�R���
�z�HC��f                                    Bxu^1�  "          @��R@$z��{�>���@\)C�� @$z��x�ÿ!G����C��                                     Bxu^@j  
�          @��@"�\�y���#�
���C���@"�\�q녿u�=C��\                                    Bxu^O  �          @�p�@.{�p�׾k��333C��@.{�e�����a�C���                                    Bxu^]�  T          @�(�@Dz��I�������C��@Dz��-p�����܏\C�@                                     Bxu^l\  �          @���@P  �!G���z���\)C��@P  �ff��(���Q�C���                                    Bxu^{  T          @�(�?���x�þ\)�G�C��q?���o\)��=q�v�HC�(�                                    Bxu^��  �          @�=q@
�H�mp����Ϳ�33C�&f@
�H�dz�}p��V{C��q                                    Bxu^�N  �          @��@!��.�R�
=���C�g�@!�����,�����C�޸                                    Bxu^��  "          @�
=@3�
�+��
=q���C�8R@3�
���/\)�z�C�Ǯ                                    Bxu^��  �          @�
=@:�H�<(���{��C�p�@:�H�p��G���G�C��                                    Bxu^�@  "          @��@$z��n{�O\)�"�RC�N@$z��Z�H��
=����C�u�                                    Bxu^��  
�          @��@?\)�@  ��33����C�p�@?\)�!G�����  C���                                    Bxu^�  �          @���@?\)�:�H��
=���HC��
@?\)���%��(�C��=                                    Bxu^�2  T          @�Q�@   �q녿���a�C��R@   �Y��� �����C�&f                                    Bxu^��  �          @��@����׽�G�����C���@����
�����\��C�!H                                    Bxu_~  �          @�
=?�(���  >��
@z=qC�e?�(����R�333�	�C���                                    Bxu_$  �          @��?�33����>�@���C�C�?�33��z�
=���
C�K�                                    Bxu_*�  �          @�=q@   ��
=��G����C���@   ��=q��{�[\)C�
                                    Bxu_9p  �          @�@(Q��}p����\�o
=C��@(Q��c33�
�H���
C�C�                                    Bxu_H  �          @�{@'�����8Q��Q�C�"�@'��\)���[33C���                                    Bxu_V�  
�          @��
@\)����>8Q�@��C�� @\)��=q�O\)��C��)                                    Bxu_eb  �          @���@\)�\)�����\)C��
@\)�o\)���
��=qC��{                                    Bxu_t  T          @�\)@���x�ÿ�z��c�C��
@���`���33���
C�4{                                    Bxu_��  "          @���@ff�n�R��  ����C�
@ff�Mp��%�=qC�
                                    Bxu_�T  �          @�  @
=q�c33��\����C��=@
=q�:=q�Dz��!  C�P�                                    Bxu_��  "          @��\@��\(��=q��C�O\@��1G��I���#\)C�J=                                    Bxu_��  T          @��@ ���j�H�z���  C�+�@ ���Dz��8Q���C��
                                    Bxu_�F  
�          @�@��s�
���ǮC�� @��Mp��:�H�\)C�H                                    Bxu_��  "          @���@�H�3�
�Fff��
C�c�@�H���R�k��F=qC�AH                                    Bxu_ڒ  
�          @�z�@�H�\�x���V��C���@�H�\)��(��l{C�~�                                    Bxu_�8  
�          @��@�
���z=q�S�C��f@�
�\(���\)�n�HC��                                     Bxu_��  T          @�
=@Q쿨���{��e�HC�\@Q쾸Q���(��y��C�+�                                    Bxu`�  �          @��\@�H�{�_\)�:��C�@�H����z�H�[  C���                                    Bxu`*  "          @��@,(��+��AG����C��3@,(���33�c�
�<�C�n                                    Bxu`#�  
�          @�p�@p���녿�(��g33C��R@p��j=q�����G�C���                                    Bxu`2v  �          @��@Q���33�\���C���@Q��hQ��(���
=C�5�                                    Bxu`A  �          @�?�z���zῐ���T��C�Ǯ?�z���  �Q����C���                                    Bxu`O�  
�          @��?������\���H�e��C��?����{�������HC�/\                                    Bxu`^h  
Z          @�@\)�z�H����{C�8R@\)�Z�H�"�\���C��                                    Bxu`m  
�          @�\)@6ff�l(��ٙ���ffC���@6ff�L(��!G����
C�޸                                    Bxu`{�  P          @�ff@B�\�o\)���Z�HC��\@B�\�W
=� �����C��                                    Bxu`�Z  &          @��
@a��*�H������C�w
@a����� ������C�c�                                    Bxu`�   
�          @�p�@S33�J�H��p����
C�
=@S33�+������C�n                                    Bxu`��  T          @�{@Y���%�=q��p�C�\)@Y�����H�<����RC�                                    Bxu`�L  �          @�33@�H��33>��
@w
=C�K�@�H��녿#�
���C�j=                                    Bxu`��  �          @��\@
=��>��@��
C��q@
=��p���\���
C��H                                    Bxu`Ә  �          @�p�@7��~�R>B�\@G�C��@7��z�H�=p��
=qC�)                                    Bxu`�>  T          @�ff@I���s33���
�o\)C���@I���g������`(�C��                                     Bxu`��  
�          @�Q�@XQ��j=q����  C�Z�@XQ��\(���{�|(�C�@                                     Bxu`��  
�          @�\)@8Q��|(��n{�,��C��@8Q��g�������  C�Ff                                    Bxua0  "          @�\)@333�{���33�V�\C���@333�c33����p�C�                                      Bxua�  �          @�
=@@  �w
=�+���33C���@@  �fff��ff���\C���                                    Bxua+|  
Z          @��R@<(��j=q��33�\��C�aH@<(��R�\��(���z�C��                                     Bxua:"  "          @�ff@:=q�/\)�:�H�p�C�\)@:=q���R�^{�2=qC�Ф                                    BxuaH�  �          @�\)@>�R�8Q��/\)���C���@>�R�
�H�U��(��C���                                    BxuaWn  �          @��@J=q�s33�b�\�;��C��@J=q�����j=q�D�RC�\                                    Bxuaf  "          @�p�@=p��ff�S33�)�C�P�@=p����
�l���C��C�N                                    Bxuat�  �          @��R@,���6ff�>�R���C��\@,����c�
�9{C�"�                                    Bxua�`  "          @��@ �����ÿ�z����C�:�@ ���^�R�2�\�(�C��)                                    Bxua�  
�          @�{@��xQ��(��ӮC�)@��QG��A��{C�@                                     Bxua��  T          @�{@ ���l���$z����HC�B�@ ���@���Vff�+G�C��                                    Bxua�R  �          @�p�?�33�k��2�\�
��C�{?�33�;��dz��:��C���                                    Bxua��  
�          @�{?�
=�3�
�dz��9ffC�8R?�
=��z���(��b��C��H                                    Bxua̞  T          @�z�@,������>�?\C��)@,���~{�O\)���C�                                      Bxua�D  �          @�(�@����\��Q���G�C�^�@���(�������33C��                                    Bxua��  
�          @��\?�����p��\���C�Y�?�����
=������\C��R                                    Bxua��  
�          @��@��#33@#33B
=C��R@��Dz�?��A�Q�C�                                    Bxub6  
�          @�{@	���޸R@���BdQ�C��H@	���+�@p��B>G�C�aH                                    Bxub�  �          @�33@=q���@�G�Bj�C�ٚ@=q�,��@���BG(�C���                                    Bxub$�  "          @��\@'����H@��B_33C�l�@'��.�R@��HB<�C��\                                    Bxub3(  
�          @��
@>�R���@��B<�C���@>�R�QG�@\��B�C�.                                    BxubA�  
�          @�p�@5��Q�@�Q�BD=qC��@5��S33@fffB�\C�J=                                    BxubPt  
�          @��H@R�\�^�R?�ffA��HC��{@R�\�s33?s33A+�C�q�                                    Bxub_  
�          @���@���녿!G���  C��@���G�����\)C�                                    Bxubm�  �          @�(�@P  �p  @�
A�{C�y�@P  ���
?��AD  C�%                                    Bxub|f  
�          @�33@l���`  ?��
A��C�Q�@l���p  ?0��@�C�L�                                    Bxub�  T          @�  @w��K�?�ffA�p�C�K�@w��\��?G�AQ�C�#�                                    Bxub��  
�          @�=q@j=q�\(�?�  A�G�C�^�@j=q�p  ?h��AG�C�#�                                    Bxub�X  S          @��H@U�p  ?޸RA��C��q@U����?W
=A��C��f                                    Bxub��  
�          @�p�@aG��}p�?��\A-C�˅@aG����H=��
?W
=C�XR                                    BxubŤ  
�          @���@g��z�H?��AYG�C�aH@g����>�33@mp�C��3                                    Bxub�J  �          @�@x���a�?�  AV�RC��@x���n�R>��@�33C�!H                                    Bxub��  �          @�
=@XQ����H?��\AW33C��f@XQ�����>���@G
=C�*=                                    Bxub�  �          @�Q�@G
=��Q�>k�@�HC�E@G
=��{�O\)�  C�y�                                    Bxuc <  �          @��@Fff��
=?\)@��C�aH@Fff��\)����Q�C�Y�                                    Bxuc�  "          @��R@`  ���>���@L��C�@`  ���
�(����ffC�(�                                    Bxuc�  �          @�ff@mp��{���p��{�C��\@mp��o\)��G��W�C�e                                    Bxuc,.  �          @�{@����fff�   ��G�C�)@����X�ÿ���`  C��
                                    Bxuc:�  
�          @�z�@s�
�e?�\)AAC�Y�@s�
�p  >�=q@8��C��R                                    BxucIz  �          @�  @%���p���&ffC�,�@%���H��
=���C�,�                                    BxucX   
�          @��R@>{��ff�0����z�C��@>{�{��У�����C��                                    Bxucf�  �          @��R@+���  ���\�b�HC�R@+��vff�p���  C�h�                                    Bxucul  T          @���?�Q���Q��������C��?�Q��~{�333��\C�                                    Bxuc�  W          @�  @*�H�'
=<��
>�z�C���@*�H�"�\�z��
=C�/\                                    Bxuc��  T          @�@��\��(�?޸RA���C��@��\��=q?�{AuC��                                    Bxuc�^  �          @��\@�ff����@��A��\C���@�ff��z�?�A��C�K�                                    Bxuc�  	�          @�=q@�Q쿆ff@'
=A���C�t{@�Q��\)@33A��HC�"�                                    Bxuc��  �          @��
@�{�J=q@�HAׅC�@ @�{����@
�HA�=qC�&f                                    Bxuc�P  �          @��H@��k�?�=qA��\C���@�����?���A���C��H                                    Bxuc��  
�          @��
@��H�!G�?ǮA��C�w
@��H�xQ�?�\)Am�C��R                                    Bxuc�  �          @���@���G�?��
A���C��@����?���Ay�C���                                    Bxuc�B  	�          @�33@��
�p��?�p�A��C���@��
��\)?��HA�\)C�*=                                    Bxud�  
Z          @���@�33���
@   A�C��\@�33���H?��HA�  C�/\                                    Bxud�  T          @���@��
���\?�p�A��C�R@��
���?���A33C��                                    Bxud%4  "          @��@��ÿ���?�AuG�C�K�@��ÿ��R?���A>=qC��)                                    Bxud3�            @�(�@%���z�>�@��C�5�@%���(��\)����C�>�                                    BxudB�  �          @��\@Q���33?xQ�A1p�C�S3@Q����R�L�Ϳ
=C��                                    BxudQ&  �          @��@0  ��{�����\��C�� @0  ��Q쿡G��g�
C�5�                                    Bxud_�  �          @�(�@Dz��z=q?n{A*=qC�\@Dz�����<#�
>�C���                                    Bxudnr  �          @���@a��xQ�>�@�  C�!H@a��x�þ�
=��33C��                                    Bxud}  �          @�z�@K���=q�����ffC�'�@K���p������C33C���                                    Bxud��  �          @�ff@p  �tz�?\(�A�RC�:�@p  �z�H�#�
�ǮC��                                     Bxud�d  T          @�z�@n�R�w
=��\)�333C�H@n�R�o\)�s33�#33C�t{                                    Bxud�
  �          @�(�@U���{���H��C�B�@U��~{��
=�w�C��                                    Bxud��  �          @�z�@A���{��\)�>{C�#�@A���Q쿥��^=qC��{                                    Bxud�V  �          @�\)@^�R�s33��p����\C�:�@^�R�g
=��p��\��C��3                                    Bxud��  �          @�p�@�  ��(�?�{AI�C�h�@�  ��
=?=p�A�C�=q                                    Bxud�  �          @�=q@�\)��z�>�ff@���C�Y�@�\)��(�=u?(��C��                                    Bxud�H  �          @�ff@��ÿ��?Tz�A{C�0�@�����>�p�@��C�y�                                    Bxue �  �          @��@�Q쿠  ?E�A�C��@�Q쿱�>�@��HC�@                                     Bxue�  �          @��\@�33��z�?(�@��C�B�@�33��G�>�=q@7�C���                                    Bxue:  �          @���@�G���
=?���A?�C�g�@�G���33?L��A
ffC�:�                                    Bxue,�  �          @��@��H��(�?fffA��C�Ff@��H���?��@�
=C�Y�                                    Bxue;�  �          @�(�@�ff���>�@��C�f@�ff��{>��?���C���                                    BxueJ,  �          @��@���)����(����\C�5�@����R���
�:�RC��                                    BxueX�  T          @���@�ff�1녿B�\�(�C��@�ff�!녿����qC�4{                                    Bxuegx  �          @�Q�@���\����8Q�C��@��
�H�G��
=qC��                                    Bxuev  �          @�=q@�����(�>\)?ǮC�� @�����(��L����C���                                    Bxue��  �          @���@��Ϳ��H�\)��(�C�j=@��Ϳ�녾����ffC��H                                    Bxue�j  �          @���@��H��aG��=qC�S3@��H��p��333��33C�޸                                    Bxue�  T          @�=q@���Fff>��
@^�RC���@���E�����C��\                                    Bxue��  �          @��\@�Q��Q�?�@�  C��f@�Q��\>�?�{C�AH                                    Bxue�\  �          @�G�@����aG�?�  A[�C��@�����33?��\A1p�C���                                    Bxue�  �          @��H@�ff�s33?޸RA�(�C��{@�ff����?�(�A�{C���                                    Bxueܨ  �          @���@�zΌ��@A�(�C�Q�@�z���?ٙ�A�C���                                    Bxue�N  �          @���@�33���?�ffA�ffC�H@�33���H?�
=AO33C�H�                                    Bxue��  �          @���@�p���
=?���A�=qC�Ff@�p���p�?��AHQ�C���                                    Bxuf�  �          @�=q@�p�>\)?�z�AI�?���@�p���?�z�AJ=qC�K�                                    Bxuf@  �          @�=q@�{>#�
?���A:ff?�\@�{��Q�?�=qA;�
C�}q                                    Bxuf%�  �          @��@��>L��?��A5p�@{@���L��?�ffA8��C���                                    Bxuf4�  �          @���@�����
?s33A&�\C��=@������?fffA�\C�T{                                    BxufC2  �          @�
=@��>�?O\)A(�@�p�@��>��?fffA\)@:�H                                    BxufQ�  T          @�  @�z�k�?\(�A33C���@�z��
=?G�A	��C���                                    Bxuf`~  �          @��@���fff?��A<��C��f@����\)?Q�A{C���                                    Bxufo$  �          @�Q�@�33���?���AD��C�w
@�33�\?B�\AQ�C�Ff                                    Bxuf}�  �          @���@��?5@1G�A��A��@��=#�
@7
=B \)?�\                                    Bxuf�p  �          @��@�?�p�@(Q�A�RA��@�?Tz�@9��B  A)p�                                    Bxuf�  �          @��@��R?�G�@p�A���Ao�@��R?5@(�A�33A	G�                                    Bxuf��  �          @�Q�@���?�R?��RA�p�@��
@���>\)@�A�  ?У�                                    Bxuf�b  �          @�
=@�33?(��?��A�
=@�  @�33>aG�?�33A���@(Q�                                    Bxuf�  �          @��@�  >�
=?���A�Q�@�=q@�  �L��@   A��\C��
                                    Bxufծ  �          @��@�33=�Q�?�(�A�ff?�ff@�33���
?�Q�A��
C�"�                                    Bxuf�T  �          @�@�(���(�?�z�A��C���@�(��L��?�G�A��C�W
                                    Bxuf��  �          @�(�@�Q�z�H?���A�{C�1�@�Q쿪=q?�=qAp��C�7
                                    Bxug�  
�          @�(�@��ÿ���?�ffAj�RC�E@��ÿ˅?s33A+33C��                                    BxugF  �          @�p�@��H��G�?��RA��C��@��H��=q?��HAX��C�N                                    Bxug�  �          @��R@�녿�\)?��Ax��C�3@�녿�33?��
A7�C���                                    Bxug-�  T          @��@�=q�:�H?�33A��C��
@�=q���?�Q�A��C���                                    Bxug<8  �          @�
=@�G����?W
=Az�C��@�G��!�>�  @0��C�j=                                    BxugJ�  �          @�{@��Ϳ�(�?5@�z�C���@��Ϳ���>�{@s33C��R                                    BxugY�  �          @��@�
=�s33?p��A((�C��{@�
=��33?0��@�C��H                                    Bxugh*  �          @��@����G�?
=q@�C��
@����=L��?
=qC�:�                                    Bxugv�  �          @�
=@����>�@�G�C��\@���
�H�u�(�C���                                    Bxug�v  �          @�\)@�p���?(�@��C��q@�p��
==L��?�C���                                    Bxug�  �          @�\)@�Q��ff?��@\C��@�Q���=#�
>�ffC��{                                    Bxug��  �          @�
=@���
=?}p�A.�RC�Ф@����\>�@��C���                                    Bxug�h  �          @��@�{�G�>�
=@�  C�\)@�{��
��\)�:�HC�#�                                    Bxug�  �          @��@����>u@,(�C���@���þ�=q�B�\C���                                    Bxugδ  �          @��@�z��ff��(�����C��3@�z����z�H�,��C�l�                                    Bxug�Z  �          @�  @�Q�����p�����C���@�Q��녿aG��C�n                                    Bxug�   �          @���@���{�����\)C�o\@���녿���9�C�j=                                    Bxug��  �          @�ff@�Q���R������C��\@�Q������\�5�C��\                                    Bxuh	L  �          @�Q�@���'
=>\@�p�C��=@���'���  �2�\C��R                                    Bxuh�  �          @���@����)��?�z�AN�RC�&f@����6ff>��H@�ffC�4{                                    Bxuh&�  �          @�=q@����G
=>aG�@
=C��@����Dz�����RC�@                                     Bxuh5>  �          @���@����'
=>�{@q�C��@����'���z��I��C���                                    BxuhC�  �          @�p�@����R=L��?�\C��=@����H�����z�C��
                                    BxuhR�  �          @�  @~{�E���=q�lz�C��@~{�+��G���p�C��q                                    Bxuha0  �          @���@dz��R�\�	�����RC��{@dz��+��7���C���                                    Bxuho�  �          @�
=@w��A�����  C���@w����A����C�@                                     Bxuh~|  �          @��H@�  �.�R��{���C�ff@�  ����p����C���                                    Bxuh�"  �          @��@��H�#33���\�Yp�C�z�@��H�
�H������C�Z�                                    Bxuh��  T          @��@�G��   ��ff��\)C���@�G��33�
=��p�C�Ф                                    Bxuh�n  �          @�
=@�  �(�����@Q�C�c�@�  �ff��Q���ffC��                                    Bxuh�  �          @�ff@�p��*=q�L���33C�%@�p���ÿ��q�C�n                                    BxuhǺ  �          @�z�@�\)�\)���H���C�!H@�\)�33��=q�8��C�                                    Bxuh�`  �          @�33@����$z��G�����C���@�����ÿ�ff�6=qC�e                                    Bxuh�  �          @�(�@��H���fff���C��q@��H��33��33�p��C�G�                                    Bxuh�  T          @��@�G��ff���H����C��@�G��
=q��ff�4��C��)                                    BxuiR  �          @�Q�@�z���������C�1�@�z��\)���
�5��C��                                    Bxui�  �          @��@���33�����g�C�s3@���\�J=q�z�C�&f                                    Bxui�  �          @��@�=q�������eC�ٚ@�=q�녿Y����
C��3                                    Bxui.D  �          @���@����p��k��!�C���@�����B�\���C�@                                     Bxui<�  T          @��@�\)���>#�
?�G�C��)@�\)�
=��
=���\C���                                    BxuiK�  �          @��@�=q�(��?�R@��C���@�=q�-p��L�;��HC��=                                    BxuiZ6  �          @��H@��   >�{@mp�C���@�� �׾�\)�?\)C���                                    Bxuih�  T          @���@�p���z�?��RA�\)C��f@�p��p�?z�HA+33C�T{                                    Bxuiw�  �          @�z�@�z����?.{@��C�0�@�z��\)>\)?��HC���                                    Bxui�(  �          @��@�(��(�=L��?
=qC��f@�(���þ��H��\)C�*=                                    Bxui��  �          @�=q@�����#�
��G�C�4{@���33�\)���C��3                                    Bxui�t  �          @��@���33�����C��
@����׿fff�z�C�k�                                    Bxui�  �          @�33@��
�G��Q��G�C��q@��
��  ����_�
C��                                    Bxui��  �          @�ff@�(���=q��Q��w\)C��3@�(����������C��                                     Bxui�f  �          @�  @����
=q�4z����\C�H�@��Ϳ���P����RC��R                                    Bxui�  �          @�G�@j=q�G��\����
C�
@j=q�����y���4�C��H                                    Bxui�  �          @��\@��Q��AG��
=C�xR@������\����C�8R                                    Bxui�X  �          @�33@�G��p��L(��
��C���@�G������hQ��!��C��R                                    Bxuj	�  �          @���@\)�"�\�E���C��H@\)�ٙ��hQ���C�|)                                    Bxuj�  �          @��@����0  �8Q�����C���@��׿����^{�ffC��                                    Bxuj'J  �          @��
@Y���p��k��%=qC��@Y�����H���A�C�XR                                    Bxuj5�  �          @��\@j=q��R�XQ����C��=@j=q�Ǯ�x���1z�C�l�                                    BxujD�  �          @�@e��#33�`���ffC�E@e�������G��7G�C��q                                    BxujS<  �          @�(�@N�R�b�\�Z=q�C�5�@N�R�$z����5p�C���                                    Bxuja�  �          @��
@P���a��XQ����C�S3@P���%�����4(�C�Ф                                    Bxujp�  �          @��
@Fff�z=q�C�
��{C�5�@Fff�AG��|(��)33C��H                                    Bxuj.  �          @���@E��dz��@  ��RC�h�@E��-p��s33�+�C�]q                                    Bxuj��  �          @���@+��Fff�S33�=qC�j=@+����~{�C��C�u�                                    Bxuj�z  �          @�  @���Z=q�K��p�C���@���   �{��B��C��)                                    Bxuj�   �          @�(�@��o\)�h���"G�C�B�@��,�����R�Sz�C�w
                                    Bxuj��  �          @���@���E���ff�<z�C��\@�ÿ����H�fp�C���                                    Bxuj�l  �          @��@&ff�2�\�{��7G�C�z�@&ff���H��  �]\)C�Q�                                    Bxuj�  T          @�Q�@#�
�5��e�-
=C�
=@#�
����{�T\)C�,�                                    Bxuj�  �          @��H@(���,���qG��3�C�+�@(�ÿ�z����\�Y{C��                                    Bxuj�^  �          @��@+��9���s33�/�\C�aH@+���������VQ�C���                                    Bxuk  �          @��
@<���Fff�g
=� ��C��@<���������GQ�C�g�                                    Bxuk�  �          @�@C�
�<(��W
=��C��@C�
���R��  �>p�C�o\                                    Bxuk P  �          @���@B�\�2�\�j�H�&�\C��
@B�\��\��Q��I�
C��=                                    Bxuk.�  �          @���@Mp��1��`���Q�C���@Mp���ff��33�@�C�`                                     Bxuk=�  �          @�(�@6ff�S33�B�\��\C�k�@6ff��H�q��5\)C��                                     BxukLB  �          @�\)@*�H�l(��B�\��HC���@*�H�2�\�xQ��4�C��                                    BxukZ�  �          @�=q@(��|(��4z�� =qC��@(��E��n�R�2��C���                                    Bxuki�  �          @�ff?�z��j�H�O\)�33C�'�?�z��.{��=q�O�HC���                                    Bxukx4  �          @���@z�H����L(���C��f@z�H�E��`  �$��C�xR                                    Bxuk��  �          @��\@��8Q��(���\C��
@�>�ff������H@�33                                    Bxuk��  T          @�33@(Q��E�&ff���C�4{@(Q��z��S�
�-\)C�P�                                    Bxuk�&  �          @���@@�׿�z��!��33C��R@@�׿�Q��;��(z�C�8R                                    Bxuk��  �          @���@��\?�G�� ����{A^=q@��\?��R��
=���HA���                                    Bxuk�r  �          @�33@b�\?����AG���A�ff@b�\?�Q��'
=��A�z�                                    Bxuk�  �          @�p�@|��?�
=��
=���
A�ff@|��@Q쿷
=���A�(�                                    Bxuk޾  �          @��@j�H?���{����A��H@j�H?�\)��ff��ffA�ff                                    Bxuk�d  �          @�G�@\��>��H�7
=�\)Ap�@\��?�(��(Q���HA�G�                                    Bxuk�
  �          @��H@Tz��R�E��)z�C���@Tz�>L���HQ��-\)@[�                                    Bxul
�  �          @��@P  �\)�Y���7Q�C�3@P  >�{�[��9ff@�\)                                    BxulV  �          @�@@  �
=�^�R�B�
C�xR@@  >�{�`���EG�@�Q�                                    Bxul'�  �          @�z�@1녾��U�G�C��@1�>���Vff�HG�Aff                                    Bxul6�  �          @��R@\)�\�tz��m�C�8R@\)?#�
�r�\�i�A~{                                    BxulEH  �          @��@�R�333�o\)�]�\C�{@�R>����r�\�bff@��                                    BxulS�  T          @��R@_\)��z���\��ffC��@_\)�B�\�%��=qC��                                    Bxulb�  �          @��@_\)�B�\?G�Az�C�|)@_\)�H�ýL�Ϳ(�C�
=                                    Bxulq:  �          @�  @mp���(���  ����C���@mp��5� ���߮C��\                                    Bxul�  �          @��\@H��=����i���E33?��@H��?���_\)�:
=A�                                    Bxul��  �          @�z�@E�>��H�p���IffAG�@E�?�(��_\)�6�
A�{                                    Bxul�,  T          @���@Q�>����i���?p�@���@Q�?���[��0A�33                                    Bxul��  �          @�Q�@l(�����U�(z�C��3@l(�?=p��P���#��A5p�                                    Bxul�x  �          @���@e����Z=q�-Q�C��q@e�>���[��.Q�@θR                                    Bxul�  �          @�G�@[�����333�(�C��R@[���\�B�\�%  C��                                     Bxul��  �          @��\@Q��L(���z��h��C��H@Q��@  �����d��C��                                    Bxul�j  �          @�=q@^�R�S33>�\)@X��C�9�@^�R�P�׿(�����C�k�                                    Bxul�  �          @���@[��P  >Ǯ@�{C�Ff@[��O\)�   ��Q�C�W
                                    Bxum�  �          @��@|���%=�?��HC�]q@|���!녿z�����C��\                                    Bxum\  
�          @�(�@n�R�)��������C�T{@n�R��H���k33C���                                    Bxum!  �          @���@k���\�=p��=qC��@k�� �׿�����ffC��\                                    Bxum/�  �          @�
=@[��1녿&ff�G�C��@[��   ��\)���C��R                                    Bxum>N  �          @�p�@=p��b�\=�?�G�C��
@=p��\(��W
=�'\)C�\)                                    BxumL�  �          @��@C�
�e�?}p�A>=qC�E@C�
�mp��#�
���C��                                     Bxum[�  �          @�{?�33�aG�@7
=B�C�+�?�33���?�ffA��C�B�                                    Bxumj@  �          @�G�@��XQ�@FffBC���@���33@z�A���C�33                                    Bxumx�  �          @�p�?��H��@S�
B)�C�1�?��z�H@�A�ffC��{                                    Bxum��  �          @�ff?��H�.�R@^{B<�\C�H?��H�dz�@%B  C��                                    Bxum�2  �          @�G�@��(�?�
=A��\C�s3@���?
=qA�RC���                                    Bxum��  �          @�G�@(���Y����
=�u�C��=@(���<����
�ܣ�C���                                    Bxum�~  �          @��@)���l(��#�
�#�
C�� @)���b�\���\�O�C�k�                                    Bxum�$  �          @�\)@.�R�s�
?�@��HC��@.�R�s�
�
=q���
C�Ф                                    Bxum��  
�          @�\)@7
=�l(�?�R@��
C�� @7
=�n{�������C���                                    Bxum�p  �          @�
=@9���QG�=�Q�?�(�C���@9���J�H�O\)�*�RC�>�                                    Bxum�  �          @�Q�@Tz��=p���R����C�!H@Tz��*�H��z���Q�C��\                                    Bxum��  �          @�z�@J�H�HQ�=p���HC���@J�H�3�
��=q��G�C�C�                                    Bxunb  �          @�G�@@  �8�ÿ�z����C��@@  �z����� ��C�                                      Bxun  �          @�\)@.{�5�ff���C��@.{����333�C��                                    Bxun(�  �          @���@<���C�
����\��C��q@<���)��������=qC��                                    Bxun7T  �          @���@Q��@  >�z�@q�C���@Q��=p������C���                                    BxunE�  �          @�=q@8���i��?xQ�A<��C�4{@8���qG����
�z�HC���                                    BxunT�  �          @���@1G����
��\)�;�C�|)@1G������ff��  C�:�                                    BxuncF  �          @�(�@<�����!G�����C��R@<�����\�������C���                                    Bxunq�  �          @�
=@J�H�������C���@J�H���
��
=��33C�Ǯ                                    Bxun��  "          @��
@z���G���33�s�
C��f@z���  ��
=��G�C���                                    Bxun�8  T          @��@.{�����=q�g
=C���@.{�w
=�!G�����C���                                    Bxun��  �          @�Q�@6ff��=q������C��H@6ff�\(��.{��Q�C��                                    Bxun��  �          @�(�@^�R�u��{�k�C�R@^�R�S33������C�C�                                    Bxun�*  �          @�@hQ��b�\��\���
C��H@hQ��8���,�����C��)                                    Bxun��  �          @�33@c�
��=q���H�rffC��
@c�
�_\)�#33��G�C��=                                    Bxun�v  �          @��@b�\�{���Q����C��q@b�\�R�\�/\)��=qC��=                                    Bxun�  �          @�{@n�R��  ������{C���@n�R�X���(����33C���                                    Bxun��  �          @��
@j=q�������R�LQ�C�@j=q�a���ŮC��)                                    Bxuoh  �          @�G�@aG���ff�0�����C��@aG��tz��{��33C�S3                                    Bxuo  �          @�G�@`�����Ϳ���.{C�"�@`���j�H�(���(�C�ٚ                                    Bxuo!�  �          @�G�@����\(������\)C�� @����4z��#33��\)C�w
                                    Bxuo0Z  �          @���@�Q��S�
��
=��Q�C�3@�Q��,(��#�
��\)C��q                                    Bxuo?   �          @��@�z��p������z�C�^�@�z����H�ׅC��q                                    BxuoM�  T          @��
@�
=�
=q�����C�p�@�
=���
����C�\                                    Bxuo\L  �          @��
@�=q��
=��
=��
=C��3@�=q��ff�����(�C���                                    Bxuoj�  �          @�Q�@�{��Ϳ�Q����\C���@�{����$z���\)C�u�                                    Bxuoy�  �          @��
@�Q��A녿�33���C�xR@�Q��ff�,�����C��3                                    Bxuo�>  �          @�33@}p��7��
=q����C�@}p����9���z�C��                                    Bxuo��  �          @�=q@~{�,(��33�̸RC��@~{��z��>�R�\)C�*=                                    Bxuo��  �          @���@�����{�0�����
C���@����z�H�K��\)C�5�                                    Bxuo�0  �          @�(�@}p���
�5�����C��)@}p�����XQ���\C�\)                                    Bxuo��  �          @�z�@�
=��z��.{��C��{@�
=���
�J=q�=qC�#�                                    Bxuo�|  �          @�z�@j=q�)���7
=��C�\@j=q��Q��`���$G�C��R                                    Bxuo�"  �          @�(�@J=q�Mp��8Q��\)C�C�@J=q�{�l(��.�HC�j=                                    Bxuo��  �          @��R@�����\�!G���\)C���@��ÿ����E��	{C���                                    Bxuo�n  �          @�{@�33�p��ff���C���@�33���9���p�C��
                                    Bxup  �          @�ff@�=q�z��%���C�8R@�=q���H�E��	��C�,�                                    Bxup�  �          @�\)@�(��ff�9��� (�C��f@�(���33�Y���ffC�9�                                    Bxup)`  �          @�\)@u�5�(����p�C�� @u��
=�W��(�C��                                    Bxup8  �          @�\)@z=q�0���*=q��z�C�^�@z=q�����W
=�ffC�W
                                    BxupF�  �          @�Q�@�ff�{�#�
�ݙ�C��@�ff�����J�H�
=C��
                                    BxupUR  �          @���@�녿�p��3�
��33C��@�녿���Q���C��                                    Bxupc�  �          @���@������E�G�C��@���aG��`���33C�
=                                    Bxupr�  �          @�  @�����5�����C���@������Tz���C��                                     Bxup�D  �          @�Q�@�����R�*=q��\C�=q@��ÿ�=q�L����C�Z�                                    Bxup��  �          @�Q�@�=q�����#�
��33C�7
@�=q�(��8Q��   C�33                                    Bxup��  �          @��H@���8Q��N�R�z�C�t{@��>�z��R�\��H@h��                                    Bxup�6  �          @��\@�
=?��8Q��(�@�@�
=?�z��$z����A��
                                    Bxup��  �          @�G�@L(�>L����\)�Z
=@hQ�@L(�?�ff���R�G��A���                                    Bxupʂ  �          @���@J�H�=p���z��U��C�aH@J�H?�R����WQ�A1�                                    Bxup�(  �          @�
=@�G���{�aG�� �C�S3@�G�<#�
�l(��)��>\)                                    Bxup��  �          @���@��
��R�aG��\)C���@��
>��b�\�z�@�                                      Bxup�t  �          @��
@�33�8Q��S�
���C�H�@�33>��
�XQ��{@�
=                                    Bxuq  �          @��@����ff�:�H���C�>�@��ÿ�{�Z�H��C�N                                    Bxuq�  T          @��
@�=q����?\)��C��@�=q�L���X����C�u�                                    Bxuq"f  �          @�@��R�ٙ��@  ��
C�f@��R�333�W��\)C�K�                                    Bxuq1  �          @��@�=q��(��H����C��@�=q�L���W
=��C���                                    Bxuq?�  �          @�
=@�=q��녿�{���RC��R@�=q������\���C�                                    BxuqNX  �          @�Q�@�z῵��33�C\)C�H�@�z�}p�������RC��f                                    Bxuq\�  �          @�G�@�\)��G��k��G�C���@�\)��33����`z�C��\                                    Bxuqk�  �          @���@�G����H�S33�{C�&f@�G��(��j=q� ��C��)                                    BxuqzJ  �          @�{@�����z��2�\��Q�C�� @����s33�P  ��
C�)                                    Bxuq��  �          @�  @�\)�����[���
C�J=@�\)=L���fff�
=?!G�                                    Bxuq��  �          @��@�{�@  �Mp��33C�xR@�{>�z��Q��@`                                      Bxuq�<  �          @�Q�@��R>��
=���@��H@��R?�Q����Q�Ac\)                                    Bxuq��  �          @��\@���#�
�W���C��{@��?���L���{AW�                                    BxuqÈ  �          @�z�@��
��\�q��)ffC�t{@��
?333�p  �'Ap�                                    Bxuq�.  "          @��@�(��^�R�^{��\C�Z�@�(�>���dz���@XQ�                                    Bxuq��  �          @���@�zΰ�
�*�H��=qC�E@�z�\�<(��{C���                                    Bxuq�z  �          @�ff@�=q�����   ���C�u�@�=q��(�� �����C���                                    Bxuq�   �          @�G�@��
��p��!�����C�!H@��
��p��2�\���C���                                    Bxur�  �          @���@�33�Y���*�H���C���@�33=#�
�333���R?�                                    Bxurl  �          @�ff@��׾��
�U��ffC���@���?@  �P�����A (�                                    Bxur*  �          @�Q�@�G���ff�1����C��)@�G���Q��C33�G�C���                                    Bxur8�  �          @��@��׿Tz��-p���{C�3@���=��
�5���?h��                                    BxurG^  �          @�  @�G��\(���\��Q�C�"�@�G��.{�p����C�
=                                    BxurV  �          @�Q�@�33�����*=q��G�C�B�@�33���?\)�G�C��                                     Bxurd�  �          @�=q@�
=�����(Q���C�5�@�
=����:�H����C���                                    BxursP  �          @�Q�@�33�˅�#�
��C�|)@�33�.{�;�� �
C��                                    Bxur��  �          @�G�@�p��aG��Mp��  C�` @�p�>L���Tz��@$z�                                    Bxur��  �          @���@�녿���S33�ffC�1�@��=�\)�]p����?n{                                    Bxur�B  �          @�G�@�G����
�/\)��(�C��
@�G��O\)�J=q��
C��                                    Bxur��  �          @�{@���.�R��
=���
C��
@�����H�-p���33C��3                                    Bxur��  �          @�@�{�������C�AH@�{����-p���Q�C���                                    Bxur�4  �          @�@��������
�ɮC�^�@��׿\�=p��p�C�8R                                    Bxur��  �          @�ff@����
=q�"�\��z�C���@�����(��Fff�{C�\                                    Bxur�  �          @�{@�p���R�z�����C�� @�p����1G�����C�3                                    Bxur�&  �          @�\)@�z�0�׿�p���33C�&f@�zὸQ��{����C�xR                                    Bxus�  �          @�{@c�
�7
=������Q�C���@c�
�ff�+���
C��q                                    Bxusr  �          @���@+���
=�  ����C�8R@+��P���`����C��                                     Bxus#  �          @�G�@QG��r�\�������C�c�@QG��8Q��P���(�C�N                                    Bxus1�  �          @�Q�@�z���\��\���C�!H@�zῡG��&ff����C�h�                                    Bxus@d  �          @�G�@�
=������  C�c�@�
=��p��(Q���ffC��3                                    BxusO
  �          @�
=@�p��?\)��Q���{C�,�@�p����4z�����C�/\                                    Bxus]�  �          @�G�@r�\�[��z����\C��@r�\�#33�E�=qC�                                    BxuslV  �          @�G�@~{�\(���(���G�C��=@~{�*�H�0����{C�                                      Bxusz�  �          @��@�z��W����
��C�k�@�z��*�H�#�
�ۙ�C���                                    Bxus��  �          @�ff@p���X�������\C��)@p���!G��B�\�ffC��                                    Bxus�H  �          @�
=@xQ��R�\���H��
=C�ٚ@xQ�����<����C��                                     Bxus��  �          @��R@q��Y�����H����C��@q��#33�>�R��HC��q                                    Bxus��  �          @��H@G
=�tz������C���@G
=�4z��^�R�z�C��                                    Bxus�:  �          @��?��H�����333��Q�C���?��H�e�����;��C���                                    Bxus��  �          @��H?�z���
=�G
=� ��C�Q�?�z��Z=q��Q��J�
C�@                                     Bxus�  �          @�Q�@�
�����B�\���C���@�
�HQ����H�D��C���                                    Bxus�,  �          @���?����\)�\����RC��?���4z����X  C��                                    Bxus��  �          @��H@$z����
�L���z�C���@$z��333����DC�Ff                                    Bxutx  �          @��@%���R�HQ��p�C��=@%�9�����
�AQ�C��)                                    Bxut  �          @��H@
=��p��<����  C�\@
=�J=q�����=(�C�e                                    Bxut*�  �          @���@#�
��(��I����C��@#�
�4z����
�C��C�!H                                    Bxut9j  �          @�=q@'��|(��Y����
C��{@'��#33��G��L�\C��=                                    BxutH  �          @��@g��w
=������=qC���@g��Fff�2�\���C��R                                    BxutV�  �          @�33@e��xQ��=q����C�T{@e��AG��B�\��C��                                    Bxute\  �          @��@_\)��Q�Ǯ��ffC���@_\)�O\)�5���  C���                                    Bxutt  
(          @��@Z=q��33��  ��
=C�޸@Z=q�P���C33�z�C�*=                                    Bxut��  �          @�=q@e��G����H�J�RC��{@e�XQ��!����C�e                                    Bxut�N  �          @��@\����G���\)�g�C�>�@\���Tz��+���C�
=                                    Bxut��  �          @�33@��
�n{����z�C���@��
�S33������=qC���                                    Bxut��  �          @��R@vff�s�
��=q�8��C���@vff�^�R�������RC���                                    Bxut�@  �          @��
@�(��QG�?c�
A�C��@�(��W������Dz�C�4{                                    Bxut��  �          @�
=@������@3�
A��C���@����B�\?�\)A�ffC���                                    Bxutڌ  �          @�@�ff�=p�@Q�A���C�h�@�ff�g
=?��HAD��C��\                                    Bxut�2  �          @�Q�@~�R�^{?���A��HC�u�@~�R�s33>�=q@333C�(�                                    Bxut��  �          @�=q@{��P  @�RA��RC�/\@{��tz�?s33AC��H                                    Bxuu~  �          @�G�@q��Z�H@ffA���C��@q��|(�?E�A z�C��                                    Bxuu$  �          @�  @���HQ�?�ffA�z�C���@���c�
?
=@���C���                                    Bxuu#�  �          @��@mp��Z=q?��HA�{C���@mp��xQ�?#�
@ٙ�C��R                                    Bxuu2p  
�          @��@j=q�c33?�ffA��C��{@j=q�|��>�(�@��C�h�                                    BxuuA  
�          @��@w
=�Mp�?ǮA�z�C�q@w
=�c�
>���@i��C��=                                    BxuuO�  �          @�@j=q�Z�H?�  A3�
C�o\@j=q�c33��\)�FffC��f                                    Bxuu^b  �          @�=q@C33�O\)�'���{C���@C33�ff�fff�0�\C��
                                    Bxuum  �          @�ff@e��aG���33���C�@e��'��AG��	{C���                                    Bxuu{�  �          @�=q@hQ�����ff��(�C�}q@hQ��p  ��33��(�C�
=                                    Bxuu�T  �          @���@`  ��G���G��S�C�y�@`  �U��'
=���HC�<)                                    Bxuu��  �          @�z�@q��xQ��ff��{C�q@q��Dz��5���C�o\                                    Bxuu��  �          @��H@h���dz�˅��=qC��=@h���1G��0����ffC�aH                                    Bxuu�F  �          @�  @Tz��c�
��z�����C�z�@Tz��(���C�
�(�C��R                                    Bxuu��  �          @�  @\(��hQ쿳33�y�C��
@\(��8���'
=����C��R                                    BxuuӒ  �          @��@Vff�{���(��T��C�8R@Vff�N�R�"�\�噚C��                                    Bxuu�8  �          @��H@Q���33�.{���C�f@Q��E���33�>p�C�L�                                    Bxuu��  �          @��H@����H�
=q����C�7
@��`���j�H�$�C���                                    Bxuu��  �          @�ff@*=q������p��qp�C�}q@*=q�|(��G
=�C���                                    Bxuv*  �          @�@{�����{�^=qC�\)@{���\�B�\���C��{                                    Bxuv�  �          @��H@,����\)���BffC�ٚ@,���\)�3�
���C��                                    Bxuv+v  �          @��@fff�Z=q��R�ԣ�C�G�@fff���c33�(�C��
                                    Bxuv:  �          @��@Y���aG��%��ޏ\C���@Y����k��'
=C��
                                    BxuvH�  �          @��H@J�H�xQ��z�����C���@J�H�1G��dz��!\)C�l�                                    BxuvWh  �          @��H@=p����\������C���@=p��:�H�n{�'\)C��3                                    Bxuvf  �          @��@K����
�z���33C�ٚ@K��E��[����C���                                    Bxuvt�  �          @�(�@N{�����H��33C���@N{�J�H�Vff�ffC���                                    Bxuv�Z  �          @�G�@Tz��g
=���
���C�J=@Tz��-p��>�R��C�b�                                    Bxuv�   �          @�Q�@z�H�#33�z���
=C�p�@z�H��\)�7
=�=qC��=                                    Bxuv��  �          @���@l���C�
�\)��  C�1�@l���G��Mp��{C���                                    Bxuv�L  �          @�z�@mp��[��\)���
C���@mp��G��e��C�G�                                    Bxuv��  �          @��
@p  �<���:=q��
=C��=@p  ��z��r�\�*��C��                                    Bxuv̘  �          @���@h���6ff�?\)�z�C���@h�ÿ���u��0ffC���                                    Bxuv�>  �          @�p�@{��<���3�
���C��{@{��ٙ��l���#33C�T{                                    Bxuv��  �          @��R@g��p  �G����
C�  @g��(���`  ���C��3                                    Bxuv��  �          @�@Z=q��  �
=q����C�7
@Z=q�:=q�_\)��
C�                                    Bxuw0  �          @�=q@dz��o\)�)�����C���@dz��\)�w
=�&=qC��3                                    Bxuw�  �          @��@qG��8Q��U��33C�E@qG���
=����7C��3                                    Bxuw$|  �          @��R@n�R�(���X���(�C�\)@n�R��
=��(��:�C�8R                                    Bxuw3"  �          @�ff@W
=�fff�7
=��
=C���@W
=�G���  �2z�C�H                                    BxuwA�  �          @���@n{�2�\�L���
Q�C���@n{���������5ffC��                                    BxuwPn  �          @�@o\)�'��Vff���C���@o\)�����\�8C�XR                                    Bxuw_  �          @���@z�H����[��{C�R@z�H���x���2�C��                                    Bxuwm�  T          @�G�@tz��{�C33�=qC���@tzῑ��p���-{C���                                    Bxuw|`  �          @�Q�@g
=�
�H�^�R�\)C�|)@g
=�0����G��?  C���                                    Bxuw�  �          @���@j�H�z��dz��!  C�H�@j�H�\)���\�>��C��H                                    Bxuw��  �          @�
=@\(���ff�r�\�1Q�C�.@\(��k���{�JQ�C�)                                    Bxuw�R  �          @�33@b�\�Ǯ�h���,�HC�)@b�\�u�}p��@�C���                                    Bxuw��  �          @�G�@tz��{�j�H�&{C���@tzὸQ���  �9ffC�T{                                    BxuwŞ  �          @���@��\���R�\�
=C���@��\��z��k��(  C���                                    Bxuw�D  �          @���@��\���/\)���C�aH@��\��=q�Dz��33C�S3                                    Bxuw��  �          @�  @�  ��Q��;�� �HC��{@�  <��
�J�H��\>�=q                                    Bxuw�  �          @�{@�  �E�� ����ffC�b�@�  >�  �'���ff@AG�                                    Bxux 6  �          @�p�@��Ϳ���=p��C�W
@���>.{�HQ����@G�                                    Bxux�  �          @�p�@�(����(Q���C��@�(��^�R�Mp��  C�3                                    Bxux�  �          @�
=@�33��R�,(���Q�C�j=@�33��  �\(���HC���                                    Bxux,(  �          @���@�p���{�AG���C���@�p�>\)�Mp��(�?���                                    Bxux:�  �          @�@�(��J=q�E�
��C��)@�(�>��I���ff@���                                    BxuxIt  �          @�  @��
�����QG��ffC��@��
�8Q��g��%\)C���                                    BxuxX  �          @�G�@L(����z�H�5��C��3@L(����H���R�XffC���                                    Bxuxf�  �          @��@s33��
=�c�
�ffC��3@s33�Ǯ�����9��C�3                                    Bxuxuf  �          @��@q녿�33�e�!G�C��H@q녾�33�����;ffC�Q�                                    Bxux�  �          @��@������0�����
C���@����G��[��  C�q                                    Bxux��  �          @��@u��
�G
=�33C�y�@u�h���qG��.C�U�                                    Bxux�X  �          @��
@[����i���+�C�c�@[���{����H
=C�33                                    Bxux��  �          @�\)@�����33���\C�#�@���33�*�H���C�B�                                    Bxux��  �          @�Q�@�
=�C�
���H��\)C�@�
=����1���33C��=                                    Bxux�J  �          @�p�@�33���=q��Q�C���@�33��p��I�����C���                                    Bxux��  �          @��
@�(��8Q��z���  C��f@�(���33�8����HC��f                                    Bxux�  
�          @��\@w��@  ����RC�R@w���
=�Fff���C��R                                    Bxux�<  �          @��
@u��AG��\)�ř�C��q@u�����P  �G�C��                                    Bxuy�  �          @��@}p��.{�
=�хC��@}p��Ǯ�O\)�Q�C�AH                                    Bxuy�  �          @�G�@z�H�7
=�ff��p�C��@z�H����Dz���RC��
                                    Bxuy%.  �          @��H@^�R�R�\��������C�P�@^�R�(��(�����RC�|)                                    Bxuy3�  �          @�Q�@l���#33�.{�	��C���@l���z�����G�C�aH                                    BxuyBz  �          @��R@~�R��(�?�Q�A�{C�~�@~�R���?�ffA�
=C�K�                                    BxuyQ   �          @�Q�@<(��@�׿�(���z�C�4{@<(���(��@  �!33C�q                                    Bxuy_�  �          @�ff?��R�Q��b�\�*�
C��R?��R�˅����s\)C��=                                    Bxuynl  �          @�
=@��C33�s�
�1Q�C�Ff@����\���R�oG�C�=q                                    Bxuy}  �          @�(�@@���.{�a��$=qC��@@�׿������\�VQ�C�0�                                    Bxuy��  T          @�p�@E��&ff�fff�'=qC��\@E��n{���H�U�HC���                                    Bxuy�^  �          @��
@>�R��\�q��4C�AH@>�R����(��]33C��R                                    Bxuy�  
�          @��@HQ�У����H�D�
C�H�@HQ�>#�
�����Z
=@?\)                                    Bxuy��  �          @�(�@Fff��z������CG�C��f@Fff=�G���33�Z
=?���                                    Bxuy�P  �          @�
=?Ǯ����@QG�Bc�C�~�?Ǯ�)��@=qB�C�1�                                    Bxuy��  �          @�33�k��z�H@�(�B�z�C�j=�k��1G�@u�BX(�C���                                    Bxuy�  �          @�  �aG��8Q�@�{B�aHC|ٚ�aG��1G�@�ffBh  C���                                    Bxuy�B  �          @��=���{@��B�=qC��{=��_\)@QG�B,�\C���                                    Bxuz �  �          @�(���G�����@�ffB��Cz\��G��G
=@�G�BPG�C�f                                    Bxuz�  �          @�?&ff��z�@�B���C�W
?&ff�c33@P��B(G�C�(�                                    Bxuz4  �          @��@
�H�333@x��B>��C��f@
�H��z�@=qA��
C���                                    Bxuz,�  �          @�z�?��,��@|��BQ=qC��?����\@   A�=qC���                                    Bxuz;�  �          @���?=p����@�  B}(�C���?=p��r�\@N{BQ�C���                                    BxuzJ&  �          @��?u� ��@�G�Bp�C���?u�l(�@S�
B#C�J=                                    BxuzX�  �          @�33?s33��z�@�
=B��HC��)?s33�]p�@g�B5�C���                                    Bxuzgr  
�          @���?녿���@���B��fC�K�?��U�@�Q�BG��C��                                    Bxuzv  �          @����G��xQ�@��B�33C��
��G��Dz�@�{B]
=C�                                      Bxuz��  �          @�
=���Ϳk�@�z�B�z�C��R�����Dz�@���B_p�C��                                    Bxuz�d  �          @�{�Q녿@  @�=qB�Q�C^�=�Q��8��@���Ba��C~�                                    Bxuz�
  �          @��k��!G�@��B�=qCV+��k��1G�@�=qBe��C{��                                    Bxuz��  �          @��R��G���ff@�33B��\CK�῁G��(Q�@�Bk��Cx޸                                    Bxuz�V  �          @�
=��33��R@��B�W
CP� ��33�1G�@�=qBb�RCw��                                    Bxuz��  �          @�����;�\)@�  B�G�CN�ÿ���"�\@�(�Bw��C��\                                    Bxuzܢ  �          @�G��!G�>�z�@��B�� C{�!G��G�@��B��C|�)                                    Bxuz�H  �          @��׿.{?z�@�B�33C� �.{�޸R@�B�L�Cx�\                                    Bxuz��  �          @�  �5>�ff@��B�L�C���5����@�33B�z�Cy                                    Bxu{�  �          @�  ��G�=�Q�@�ffB��C(^���G����@��B�� C�Q�                                    Bxu{:  �          @����\)>W
=@��RB�Q�B����\)�ff@�G�B�ǮC�                                    Bxu{%�  �          @�Q쿸Q�?У�@���B��CaH��Q��@���B�{CHG�                                    Bxu{4�  �          @�  �\)�:=q@��\B^��C����\)���R@-p�A�=qC�e                                    Bxu{C,  �          @�  ��Q��K�@���BQ=qC����Q���(�@�HA�C��f                                    Bxu{Q�  �          @�Q�(��h��@l��B4{C�7
�(����?�A�Q�C�ff                                    Bxu{`x  �          @�녾�ff���@%A��\C����ff��  ?�@���C�k�                                    Bxu{o  �          @���5�@�33B��C{.�5�|��@`��B$C���                                    Bxu{}�  �          @��ÿ�R��@���B��HC{^���R�o\)@j=qB/�C�N                                    Bxu{�j  �          @���:�H��\@��B�Cw�=�:�H�j�H@i��B133C�\)                                    Bxu{�  �          @��׿8Q��z�@�33B�L�Cy@ �8Q��r�\@e�B+(�C��R                                    Bxu{��  �          @�=q�L�Ϳ�z�@�
=B�.CtJ=�L���g�@q�B633C�                                    Bxu{�\  �          @���Ϳ���@���B��RCsh�����P  @���BR(�C�/\                                    Bxu{�  �          @�����?��@�\)B�G�C	�����޸R@��RB�Q�Cz��                                    Bxu{ը  �          @��ÿ�
=>�33@��HB���C#=q��
=��{@�  B�L�Cm�H                                    Bxu{�N  �          @�(��u>W
=@�ffB���C'�\�u��\@�G�B��Ct�\                                    Bxu{��  �          @�p������R@��B���Cw쾅��0��@�33Bf{C�Ff                                    Bxu|�  �          @�����
=@�G�B��Ch&f���,��@���Be�C�.                                    Bxu|@  �          @�
=�E���{@�z�B�W
CL{�E��!�@��Bn33C}                                    Bxu|�  �          @��R�Tz��@��HB���CQ���Tz��'�@�z�Bg�\C|^�                                    Bxu|-�  �          @�  ����\)@�z�B��=C;�=����ff@�=qBr�\Cu�R                                    Bxu|<2  �          @�z�h��>���@���B���C.�h�ÿ�@�B�aHCt��                                    Bxu|J�  �          @�(��Tz��@�ffB��CQ�q�Tz��$z�@�  Bf(�C|&f                                    Bxu|Y~  �          @�ff�n{���R@�33B�33CF0��n{�\)@��RBl�HCys3                                    Bxu|h$  �          @�
=�Q녾�@�(�B�
=CQO\�Q��!�@�{Bf(�C{��                                    Bxu|v�  �          @�  ��녿��@���B�u�C|L;���P  @p��BCC�b�                                    Bxu|�p  �          @�{��(���(�@�ffB��\C}����(��^�R@s�
B=��C�p�                                    Bxu|�  �          @�p������&ff@�B�
=CPLͿ����/\)@��BY(�CvB�                                    Bxu|��  �          @�(����R����@��RB���CYLͿ��R�L(�@}p�BA��Ct�                                    Bxu|�b  �          @����(��Y��@�p�B�W
CV���(��:�H@���BP{CwO\                                    Bxu|�  �          @�{����>\)@�=qB�ǮC/�q�����ff@��
Bh�RCc\                                    Bxu|ή  �          @�(���p���ff@�=qB��CB����p��(Q�@��BX��Cl�H                                    Bxu|�T  �          @�G�������@��RB���CZ녿����U@��BC\)Cv��                                    Bxu|��  �          @�Q쿦ff�}p�@�Q�B�aHCY=q��ff�L��@���BLffCw�
                                    Bxu|��  �          @��ÿ�  �G�@��
B�\)CY�q��  �Dz�@�\)BX�
C{��                                    Bxu}	F  �          @�\)��(�����@�Q�B�\)CQ���(��U�@�
=B>
=Cos3                                    Bxu}�  �          @�  �33��33@��B���CQB��33�Vff@�{B;�Cnu�                                    Bxu}&�  �          @��	����ff@���B��CM���	���N{@���B<{Cl+�                                    Bxu}58  �          @�녿�z�h��@�z�B�aHCM����z��E@��RBD��Cn^�                                    Bxu}C�  �          @��ÿ�\)�fff@�(�B�  CM�f��\)�E�@�ffBE�\Cn�                                     Bxu}R�  �          @��\��
=��p�@�p�B�33CX8R��
=�Y��@��HB<�RCs�R                                    Bxu}a*  �          @��\��\)��ff@�p�B�G�CZǮ��\)�]p�@���B:Ct�R                                    Bxu}o�  �          @�Q쿼(���\)@�(�B�8RC_쿼(��`��@~�RB9(�CwQ�                                    Bxu}~v  �          @�G�������
@�(�B�G�Co�\����xQ�@s33B-��C\                                    Bxu}�  �          @�����\@�
=B��C{#׿��l(�@\)B;�C��q                                    Bxu}��  �          @���  ��@��RB�(�C��\��  �fff@���B@�RC��q                                    Bxu}�h  �          @��
�h�ÿ}p�@�{B��HCcuÿh���Mp�@�ffBN  C~(�                                    Bxu}�  �          @�Q쿋����
@��RB�u�Ce�쿋��W�@w
=B=C|�                                    Bxu}Ǵ  �          @�녿�{��ff@�ffB�� CU0���{�J=q@|��B@=qCs�                                    Bxu}�Z  �          @�  ��{��\)@��B�\CW  ��{�L(�@uB<=qCsB�                                    Bxu}�   �          @�����ÿ�z�@���B���CX�������S33@~{B=p�Ct��                                    Bxu}�  �          @����33��(�@��HB�.CXs3��33�XQ�@�  B;
=Ct                                    Bxu~L  �          @��޸R��=q@�
=B��CYff�޸R�Z�H@u�B3�
Cs�                                    Bxu~�  �          @�Q��녿�p�@�G�Bq��CU���^�R@fffB#z�Cl�\                                    Bxu~�  �          @�{��zῃ�
@�
=B��RCPc׿�z��J�H@}p�B;��Cn�H                                    Bxu~.>  �          @��
�����@��B��CT\���L(�@~{B?Q�Crc�                                    Bxu~<�  �          @�  � �׿=p�@���B�(�CH+�� ���<��@�p�BE�HCk�                                    Bxu~K�  �          @�\)�8Q�u@��Bap�CFc��8Q��:�H@hQ�B%��Can                                    Bxu~Z0  �          @�\)�*=q�\(�@�{Bl�
CE�
�*=q�9��@r�\B/Cch�                                    Bxu~h�  �          @�
=��p��E�@��B�Q�CK޸��p��@��@�ffBI�RCp\                                    Bxu~w|  �          @��R����@  @�B�G�CM������A�@���BM�
Cr��                                    Bxu~�"  �          @�녿��׾#�
@��B�  C;�f�����\)@���Bk(�Cu��                                    Bxu~��  �          @����Q�<�@�G�B�ǮC2� ��Q��   @�{Br�Ct�
                                    Bxu~�n  �          @��Ϳ�\)�8Q�@���B�ǮC={��\)�,(�@��HBkCwY�                                    Bxu~�  �          @�  ����aG�@�z�B�ǮC?�����*=q@�{Bj��Cx�q                                    Bxu~��  �          @��R�Y���u@��
B�L�CC� �Y���*�H@�p�Bk�C|Q�                                    Bxu~�`  �          @���녾aG�@�{B��CI���,(�@�\)Bo33C��q                                    Bxu~�  �          @��R�@  ����@��
B���CK� �@  �0��@��
Bi  C~                                    Bxu~�  �          @���ٙ��c�
@�\)B��fCO�H�ٙ��E�@���BC
=Cq!H                                    Bxu~�R  �          @�ff���
��Q�@��HB���C8�H���
�"�\@�{Bn��Cw�\                                    Bxu	�  �          @�G���=q�  @��B^��Cf޸��=q�|��@.�RB 33Cu(�                                    Bxu�  �          @��H�����G�@��B��CP\)�����A�@�(�BKffCtn                                    Bxu'D  �          @�=q���G�@�
=B�L�CL޸���?\)@���BG33Cp�q                                    Bxu5�  �          @��
���H��{@�Q�B��qCY+����H�S�
@{�B=��Cv�                                    BxuD�  �          @�ff�����@�z�B���Ch�)����h��@y��B6�HC}c�                                    BxuS6  �          @�{��녿�@���B�#�CE� ����6ff@�=qBR�HCp.                                    Bxua�  �          @�  ���þ�=q@�\)B�#�C=�׿����*�H@�  B]��Co�
                                    Bxup�  �          @��R���R�Tz�@�p�B��=CQ33���R�H��@�{BI\)Ct��                                    Bxu(  �          @�\)�У׿=p�@�p�B�CLT{�У��C�
@�\)BJ��Cq�                                    Bxu��  �          @�\)��{����@�
=B�C?�ÿ�{�.�R@��RBZ�Cop�                                    Bxu�t  �          @�\)��33�\)@�{B�  CF���33�:=q@��\BQ=qCp��                                    Bxu�  �          @�Q��녿��\@�z�B��CS�{����S33@�=qB?p�Cs�)                                    Bxu��  �          @�\)���H���@��HBx��C[�����H�n{@`��B��Cr&f                                    Bxu�f  �          @�{����p�@�p�B���C~���^�R@8Q�Bp�C�1�                                    Bxu�  �          @��6ff�
=@�=qBe��C?� �6ff�*�H@p��B/�C_�                                    Bxu�  �          @�\)�H�þ�  @�\)B[��C8�\�H���z�@uB233CX��                                    Bxu�X  �          @��\�J=q=#�
@�33B^  C3Y��J=q�Q�@�=qB;�RCU��                                    Bxu��  �          @��H�!G�>�z�@��RB{p�C-^��!G��z�@�Q�BXp�C[c�                                    Bxu��  �          @�p��
�H?��
@�33B~�
C���
�H����@�Q�Bv�CTxR                                    Bxu� J  �          @�z��=q?W
=@��B�\Cuÿ�=q��{@���B|{C]ff                                    Bxu�.�  �          @�p���\)>aG�@���B���C(�H��\)��\@�Q�BwG�Cs��                                    Bxu�=�  �          @��R�Tz�?�=q@���B�  B��)�Tz���@�p�B���Cq�H                                    Bxu�L<  T          @�(��Y��?�\)@�{B��3B�33�Y����(�@�33B�G�Co�R                                    Bxu�Z�  �          @���!G�?�=q@���B�aHB�Ǯ�!G����H@���B���Crs3                                    Bxu�i�  �          @�Q�˅>\@��B�u�C&���˅��=q@�{Bo�
Cd��                                    Bxu�x.  �          @������=u@�  Bx33C2������@\)BNQ�C]�                                    Bxu���  �          @����(�?k�@�
=B�C�f��(����@��
BvG�CU�\                                    Bxu��z  �          @��H�  >���@�Q�B}  C+���  ����@��
B[
=C[}q                                    Bxu��   �          @�Q��Q�>��H@��\BxG�C(:��Q�޸R@�Q�B]\)CXB�                                    Bxu���  �          @�z��G�?(��@�p�B���C!޸�G����@�p�Bm��C[�                                    Bxu��l  �          @�(���?�
=@�p�Bo�\C� �녿���@�ffBr=qCM#�                                    Bxu��  �          @�Q��
=?u@�{BtQ�C}q�
=��z�@�z�Boz�CP                                    Bxu�޸  �          @�z�@��fff@r�\Bl�C�G�@��'
=@9��B$�C�O\                                    Bxu��^  �          @�
=@��H�#33?�@�C��\@��H�\)�E����RC���                                    Bxu��  �          @�
=@�z��(�?�@�p�C�K�@�z���ÿ8Q���RC���                                    Bxu�
�  
�          @�\)@���� ��=�G�?���C��@�����׿�{�2{C�.                                    Bxu�P  �          @�=q@�ff�'�=#�
>�
=C���@�ff�z῜(��Ap�C��R                                    Bxu�'�  �          @��H@�  �"�\>�?�ffC��@�  ��\�����.{C�<)                                    Bxu�6�  �          @�{@�
=�{?   @���C�}q@�
=���!G����HC��                                    Bxu�EB  �          @�  @���  ?+�@׮C�g�@����\���H��z�C�/\                                    Bxu�S�  �          @��H@�
=�#33?�\@�G�C���@�
=��R�G���C�G�                                    Bxu�b�  �          @���@�
=�,��?�\@�{C�XR@�
=�'
=�Y���G�C���                                    Bxu�q4  �          @�{@�ff�5�?�@�G�C��)@�ff�.�R�h���z�C�(�                                    Bxu��  �          @��@�z��5?�R@�Q�C��@�z��2�\�Tz����C�                                    Bxu���  �          @�p�@��\�8��?fffA  C�/\@��\�=p��(���p�C�޸                                    Bxu��&  �          @��@�\)�>�R?L��@��HC���@�\)�?\)�=p����C��H                                    Bxu���  �          @�(�@��Fff?&ff@��
C���@��AG��n{�=qC�@                                     Bxu��r  �          @��@�33�N{?uA�C�5�@�33�Q녿:�H���C��R                                    Bxu��  �          @�@��I��?s33A\)C��3@��Mp��333����C�o\                                    Bxu�׾  �          @��
@��\�P  ?0��@׮C�H@��\�K��z�H�C�P�                                    Bxu��d  �          @�(�@�  �G
=?�p�Ai�C�aH@�  �\(��aG���C��                                    Bxu��
  �          @���@���Y��?��HA=p�C��3@���c33�����=qC�S3                                    Bxu��  �          @�p�@����Z=q?��RAA�C��@����e��z���(�C�4{                                    Bxu�V  �          @�ff@��R�aG�?L��@�G�C��q@��R�]p���G��G�C��{                                    Bxu� �  �          @�(�@��\�C33?�{AV=qC��q@��\�U������<(�C���                                    Bxu�/�  �          @��@�z��<��?�(�Ag\)C�y�@�z��R�\������RC�
=                                    Bxu�>H  �          @�p�@��Tz�?��AMG�C�Q�@��b�\�����RC�u�                                    Bxu�L�  �          @��@��R�P��?��AYC���@��R�aG��\�o\)C���                                    Bxu�[�  �          @���@�
=�0  @��A�33C���@�
=�^{?�@�G�C��{                                    Bxu�j:  �          @��@�=q�P  @
=A�ffC���@�=q�\)>�@�  C��=                                    Bxu�x�  
�          @�p�@����Fff@�RA��RC��f@����s33>�
=@���C�                                      Bxu���  �          @��@���L��@�A�\)C��@���z=q>��@�  C�<)                                    Bxu��,  �          @��@��H�5�@�A���C�+�@��H�g
=?#�
@�
=C���                                    Bxu���  �          @�z�@�\)�   @(�A�Q�C�3@�\)�W�?fffAz�C�<)                                    Bxu��x  �          @�z�@����'
=@
�HA��C��
@����U?(�@��C���                                    Bxu��  �          @���@���!�@(�A��C���@���Y��?^�RA(�C�(�                                    Bxu���  �          @��
@���/\)@  A�33C��f@���_\)?(�@��C���                                    Bxu��j  �          @��\@�
=�Q�@�HA�33C���@�
=�QG�?n{A�
C���                                    Bxu��  �          @��@�33��p�@@  A�33C�\)@�33�L��?У�A��C���                                    Bxu���  �          @�(�@�p���  @E�A�p�C�� @�p��B�\?�A�{C�y�                                    Bxu�\  �          @�z�@������@G
=A��C��q@���9��?�A��HC�H�                                    Bxu�  �          @�
=@��H���@{A��C���@��H�Vff?p��Az�C���                                    Bxu�(�  �          @���@���E�?���A��HC���@���h��>.{?˅C�l�                                    Bxu�7N  T          @���@�p��QG�?�
=A��C�|)@�p��s33=u?
=qC�l�                                    Bxu�E�  �          @�Q�@���a�?ٙ�A�p�C�"�@���y����=q�'�C��{                                    Bxu�T�  �          @��H@�(��0  @z�A�  C�L�@�(��b�\?&ff@�(�C�f                                    Bxu�c@  �          @��@�{�:=q@{A��C���@�{�hQ�>�@�ffC���                                    Bxu�q�  �          @���@����qG�?�{AMG�C�u�@����|�Ϳ0���ϮC��\                                    Bxu���  �          @�@�=q�y��?�AU�C�� @�=q��33�5���HC�3                                    Bxu��2  �          @�(�@���n�R?�G�Ae�C���@���\)������
C���                                    Bxu���  �          @�33@����g
=?�A���C��3@������H�\)��ffC��3                                    Bxu��~  �          @Å@�Q��n�R@33A���C�l�@�Q���(�>�?�p�C�!H                                    Bxu��$  �          @�33@�z��>�R@\)A�=qC��@�z��u�?+�@ʏ\C�@                                     Bxu���  �          @��
@��9��@)��A��
C�{@��u?Y��@��RC�L�                                    Bxu��p  �          @�(�@�\)�1�@0  A���C��@�\)�q�?}p�AQ�C���                                    Bxu��  �          @��@��׿�(�@<��A�C�B�@����K�?���Al(�C��=                                    Bxu���  �          @�p�@�\)�`  @ffA�C��q@�\)���R>�\)@'�C�e                                    Bxu�b  
�          @�G�@����%�@5A�RC�� @����j=q?�z�A2{C���                                    Bxu�  �          @���@����@��@"�\A�\)C�E@����xQ�?.{@���C���                                    Bxu�!�  �          @�Q�@��H�A�@4z�A�RC���@��H��G�?k�Ap�C��f                                    Bxu�0T  �          @���@��
�]p�?ٙ�A�G�C��)@��
�u��=q�"�\C�"�                                    Bxu�>�  �          @\@��
�U@A�33C�
=@��
�|(�>�?���C��                                     Bxu�M�  �          @��H@����]p�@	��A�C�E@������\=�?�\)C���                                    Bxu�\F  �          @���@��\�G�@
=A�(�C�޸@��\�xQ�>�@�p�C��)                                    Bxu�j�  �          @��@�G��6ff@(��A��C��\@�G��r�\?Y��A=qC�                                    Bxu�y�  �          @�Q�@����   @*�HAҸRC�+�@����`  ?��A�
C���                                    Bxu��8  �          @\@�33��
@�A���C���@�33�J�H?Y��A Q�C�3                                    Bxu���  �          @��H@��\�'
=?�(�A�=qC�e@��\�P  >\@hQ�C���                                    Bxu���  �          @�(�@�Q��@��?Tz�@���C��@�Q��AG��L�����C��                                    Bxu��*  �          @��H@����W
=?Q�@��C��@����S�
�}p��p�C�G�                                    Bxu���  �          @���@�
=�aG�?W
=@��
C�Y�@�
=�]p���ff��C��3                                    Bxu��v  �          @�=q@����s33?�ffAF�HC�  @����{��L�����HC��                                     Bxu��  �          @\@���ff?���AMp�C�l�@������u��C��                                    Bxu���  �          @�=q@u���Q�?��\A@��C�.@u����ÿ�Q��6=qC��                                    Bxu��h  �          @�  @�(��z�H?�G�A��C�=q@�(����þ������C��                                    Bxu�  �          @�{@tz���?��RAEC�33@tz���\)��ff�'\)C�f                                    Bxu��  �          @��@u��tz�@	��A�p�C���@u���(���Q�Y��C��{                                    Bxu�)Z  �          @�p�@dz���=q?���A�{C��H@dz����Ϳ!G��ÅC��q                                    Bxu�8   �          @��@hQ����
?�\)AXz�C��q@hQ����R����$��C���                                    Bxu�F�  �          @�z�@X�����׿   ��(�C���@X���vff�7
=���HC��=                                    Bxu�UL  �          @��R@�Q���33��Q�Y��C��@�Q��_\)�
=q����C�\)                                    Bxu�c�  �          @�ff@�\)�H��?ٙ�A��C�33@�\)�dz�\)���C�xR                                    Bxu�r�  �          @��@c�
��?�G�AuC�33@c�
����Tz���\C��
                                    Bxu��>  T          @�
=@��
�:�H?���A�z�C�� @��
�\��=��
?@  C�Y�                                    Bxu���  �          @�Q�@��]p�?�G�Ah��C���@��p  �����
=C��                                    Bxu���  �          @���@��aG�?�AYC�� @��p  �z�����C���                                    Bxu��0  �          @\@�\)�L��?���AL��C���@�\)�[������C���                                    Bxu���  �          @��
@��R�>{?�\)A)�C�%@��R�HQ�����ffC��H                                    Bxu��|  �          @�=q@�ff�=p�?n{A��C�+�@�ff�AG��5�أ�C���                                    Bxu��"  �          @��@�ff�<(�?h��A
�\C�@ @�ff�?\)�8Q����
C�
=                                    Bxu���  �          @�=q@��\�E?�A2�HC�W
@��\�P  ����p�C��=                                    Bxu��n  �          @�=q@���J�H?���AIG�C��=@���Y�����H����C��                                     Bxu�  �          @��@�  �N�R?���A+�C��{@�  �Vff�0����Q�C�R                                    Bxu��  �          @�=q@���n�R?�(�Aap�C�(�@���|�Ϳ&ff�ǮC�U�                                    Bxu�"`  �          @��H@��R�hQ�?޸RA���C�n@��R��Q쾽p��`��C�f                                    Bxu�1  �          @�33@�ff�Y��?h��A	�C��H@�ff�X�ÿxQ��=qC�Ф                                    Bxu�?�  T          @��
@����L(�?���AG33C��q@����Z=q��\��  C���                                    Bxu�NR  �          @��
@�{�`��?&ff@��
C�J=@�{�Vff��  �=�C��                                    Bxu�\�  �          @�33@�p��L��>�ff@�ffC�%@�p��>{��G��>�HC��                                    Bxu�k�  �          @��
@����L��?���At  C�o\@����c�
��\)�'�C��q                                    Bxu�zD  �          @�33@����Vff?�G�A�(�C�w
@����q녾aG��33C���                                    Bxu���  �          @�=q@�
=�Mp�?��
AC
=C���@�
=�Z=q�\)���C��                                     Bxu���  �          @\@��H�>{@�
A�G�C�8R@��H�g
=>k�@��C��f                                    Bxu��6  �          @���@���5�@�A�
=C��@���`  >��
@B�\C�C�                                    Bxu���  T          @�G�@�(��>�R@�RA��C��)@�(��u�?z�@���C�33                                    Bxu�Â  �          @��@�=q�7
=@�
A�
=C��@�=q�h��?   @�Q�C�|)                                    Bxu��(  �          @\@��\�A�@
=A��\C��3@��\�k�>u@33C�T{                                    Bxu���  T          @\@�\)�fff?�
=A���C��R@�\)���\�W
=����C��3                                    Bxu��t              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxu��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxu��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxu�f  :          @�=q@�=q�fff?��
AE�C���@�=q�p  �E���RC�Q�                                    Bxu�*  "          @�G�@��@  @"�\A�\)C��\@��x��?�R@���C�`                                     Bxu�8�  
�          @�=q@�ff�x��>��@�ffC���@�ff�dz��{��ffC�Ф                                    Bxu�GX  �          @���@�Q��j�H?��An{C�q�@�Q��|(��z�����C�q�                                    Bxu�U�  �          @���@�=q�j�H?�z�A3�C��q@�=q�o\)�h����C�XR                                    Bxu�d�  �          @���@�����?�z�A�  C��@�����{�@  ��C��                                    Bxu�sJ  T          @���@{���z�?���A��C��)@{���  ����p�C���                                    Bxu���  �          @\@����q�?���AIG�C��@����z=q�W
=��{C��
                                    Bxu���  T          @\@��
���?G�@���C��\@��
��(������u�C�t{                                    Bxu��<  �          @���@�G��}p�=��
?G�C�t{@�G��[����R��  C�z�                                    Bxu���  T          @���@�33���
�����  C���@�33�\���/\)����C��3                                    Bxu���  
�          @�G�@��
��Q쿈���#\)C���@��
�7��@  ���C�XR                                    Bxu��.  "          @�=q@�z����
�333��(�C�g�@�z��I���/\)���C�#�                                    Bxu���  
�          @�z�@�{�y��?�\@�Q�C�"�@�{�e�˅�q��C�E                                    Bxu��z  T          @��@�\)�b�\��\)�u�C���@�\)���N{���C��f                                    Bxu��   	�          @�p�@�
=�r�\���
�
=C�� @�
=�,���7
=�ݮC�"�                                    Bxu��  
�          @�(�@��
�j=q�����(�C��\@��
�8Q��33���C��
                                    Bxu�l  S          @���@�=q�XQ�?8Q�@�ffC�'�@�=q�P�׿�33�,��C���                                    Bxu�#  "          @�@�(��Z�H>��?�{C�,�@�(��?\)��33�y�C��H                                    Bxu�1�  
�          @Ǯ@����S33��
=�xQ�C��@����%����C���                                    Bxu�@^  
�          @�\)@�
=�XQ쾮{�J�HC��=@�
=�,����
��=qC�N                                    Bxu�O  "          @�p�@����L�;�33�P��C�` @����"�\��(����RC�)                                    Bxu�]�  T          @�{@�=q�H�þ�33�Q�C���@�=q�\)��Q���=qC�q�                                    Bxu�lP  �          @ƸR@����P�׾\�`  C��@����%����z�C���                                    Bxu�z�  
�          @�  @���K������j=qC���@���   �   ��p�C�~�                                    Bxu���  �          @�  @�p��g
=�33����C�#�@�p�� ���hQ��33C�Z�                                    Bxu��B  �          @�G�@�
=�W��������\C��@�
=��
=�S�
���RC�g�                                    Bxu���  "          @���@�=q�aG��(���p�C�0�@�=q��p��z�H�  C���                                    Bxu���  "          @ə�@�ff�\���1G���C��@�ff���R��p��&z�C��R                                    Bxu��4  
�          @�  @��\(��G���G�C�C�@���ff���R�6  C�U�                                    Bxu���  "          @ȣ�@g
=�R�\�s�
��
C��)@g
=�O\)��
=�U33C���                                    Bxu��  
�          @Ǯ@7��Y�����H�1(�C�{@7��!G���\)�w33C�ٚ                                    Bxu��&  
�          @�{@N{�J�H��
=�,�C��)@N{���H��Q��hC���                                    Bxu���  
(          @�@��
=��=q�=C��\@?�p���(�
=Bff                                    Bxu�r  "          @�ff@������z��a�HC���@��?.{���
W
Aw
=                                    Bxu�  
�          @�@���<����33�OffC��\@��<��
���?�                                    Bxu�*�  
Z          @�z�@,���G������>p�C�l�@,�;�z�����~�C��{                                    Bxu�9d  "          @�z�?��R�>{��
=�Y33C��=?��R=��
��G�\@33                                    Bxu�H
  
�          @Å?��R�?\)��z��V�C��3?��R<��
��\)��>�ff                                    Bxu�V�  
�          @Å?�
=�����j�H�33C�0�?�
=����  8RC��3                                    Bxu�eV  
�          @�z�@33�Mp����
�E�HC��\@33���R����C�*=                                    Bxu�s�  
�          @�z�@0  � ������RG�C���@0  >�
=��ff�{�A
{                                    Bxu���  T          @�(�@A��!G���{�G�C��@A�>��
��=q�p�
@��                                    Bxu��H  T          @��H@*=q�Fff��Q��?\)C�E@*=q��=q��
=���C��                                    Bxu���  T          @�=q@H���P  �����'  C��)@H�ÿ&ff��z��hG�C�(�                                    Bxu���  �          @�=q@P���H����  �%�HC��@P�׿�����c�C��                                    Bxu��:  
�          @�Q�@R�\�n{�@  ����C��@R�\�˅��  �K��C�q                                    Bxu���  "          @�Q�@mp����������=qC�ff@mp��*�H�w
=� �\C�                                      Bxu�چ  �          @Å@q��w��0  �ׅC�(�@q녿������
�8p�C���                                    Bxu��,  
�          @Å@�=q�\(��<���癚C��@�=q������=q�4�C���                                    Bxu���  
�          @�(�@�=q�b�\�3�
���C�}q@�=q�����  �1Q�C��=                                    Bxu�x  U          @���@���p���"�\��=qC��f@����{��z��*(�C���                                    Bxu�  
�          @���@��R��  �G�����C�>�@��R��tz��C�~�                                    Bxu�#�  T          @Å@\)���\����z�C�(�@\)�Q��z�H� C���                                    Bxu�2j  
(          @���@y���u�������C��R@y��� ������+  C�b�                                    Bxu�A  "          @�p�@��\�w���
=��33C��@��\�G��k����C�*=                                    Bxu�O�  �          @Å@�Q��s�
�	�����HC��@�Q��
=�u����C��3                                    Bxu�^\  T          @ƸR@����vff�z���p�C���@�����
��  � �\C�&f                                    Bxu�m  �          @�G�@����tz��#�
���C��@��ÿ�33��{�'{C��                                    Bxu�{�  "          @�Q�@��R�N{�>{��\C�
@��R��
=��
=�)�RC���                                    Bxu��N  �          @�=q@����X���1G���{C��f@�������z��#�HC�O\                                    Bxu���  "          @��H@�{�X���%����C��@�{�\�\)��HC��                                    Bxu���  
�          @�33@���e�,(��ȏ\C��@���У����$��C�H                                    Bxu��@  �          @˅@�  �b�\�1G���ffC��@�  ��ff��
=�&�C��                                    Bxu���  T          @�(�@��H�Z�H�5���ffC���@��H��z���ff�$��C�q�                                    Bxu�ӌ  T          @�(�@�
=�aG��7���  C��f@�
=��p���G��)ffC���                                    Bxu��2  
�          @˅@����L(��.{�ʸRC�!H@��ÿ��
��  ��RC��                                    Bxu���  T          @�G�@����S33�1G��ѮC��@����������H�#�C�                                    Bxu��~  �          @�  @r�\�U�l(��ffC�O\@r�\�c�
��z��M�HC�]q                                    Bxu�$  �          @�ff@e��N�R�y����
C��q@e��0�������W�HC��{                                    Bxu��  
�          @�ff@hQ��J�H�w��33C�p�@hQ�&ff���R�U�\C��\                                    Bxu�+p  �          @�ff@|(��;��p�����C��\@|(���\����H
=C�T{                                    Bxu�:  
�          @ƸR@xQ��@  �s�
��\C�%@xQ����=q�K�RC�{                                    Bxu�H�  
�          @�\)@^�R�333���\�033C��{@^�R��������_C�/\                                    Bxu�Wb  "          @�\)@n{�5���H�$\)C�L�@n{�����
=�Tz�C���                                    Bxu�f  
�          @�{@Y���2�\��33�2�HC�Y�@Y����\)��p��b�C�n                                    Bxu�t�  
�          @�@QG��{���
�A�\C�|)@QG�>�{��\)�g33@��                                    Bxu��T  
Z          @ƸR@j=q�����6�\C���@j=q>�Q�����V��@�p�                                    Bxu���  
�          @�ff@|(�����=q�0z�C�
@|(�?�\�����H@�{                                    Bxu���  T          @ƸR@:�H�XQ�����0  C�ff@:�H�
=��{�u\)C�=q                                    Bxu��F  U          @�\)@Vff�C�
��Q��,��C��\@Vff��33��\)�d�HC��                                    Bxu���  
�          @�=q@Q��N{����,ffC���@Q녾�����i  C���                                    Bxu�̒  �          @˅@^�R�?\)��z��.�RC��=@^�R�k�����b��C�"�                                    Bxu��8  
�          @ȣ�@R�\�*=q���
�>  C���@R�\>8Q����\�h��@G�                                    Bxu���  �          @�33@;��9�����\�E�C���@;�=�Q���(��y�
?�
=                                    Bxu���  �          @�{@9���;���{�H��C�\)@9��=�����|�R@
=                                    Bxu�*  !          @�p�@H���7���Q��@C���@H��=�\)����r{?�{                                    Bxu��  �          @���@?\)����=q�S�C�xR@?\)?�R���H�u\)A:ff                                    Bxu�$v  �          @�Q�@9���z���Q��U��C��@9��?.{��\)�u��AS
=                                    Bxu�3  �          @��@W���
����L�RC�Ff@W�?W
=��  �b33A`z�                                    Bxu�A�  �          @ʏ\@p  � ����{�?�C��q@p  ?E������S\)A8��                                    Bxu�Ph  �          @�Q�@Z�H�=q�����@�C�c�@Z�H>����
=�bz�@�G�                                    Bxu�_  �          @���@\(��G�����AffC���@\(�?&ff��(��Y=qA,Q�                                    Bxu�m�  �          @�(�@}p���
�s�
�!�C�G�@}p�>����=q�=�@tz�                                    Bxu�|Z  �          @�@xQ�����tz��!��C��H@xQ�=�����{�Cff?���                                    Bxu��   �          @���@dz��
=��G��D
=C�j=@dz�?xQ�����P�As�                                    Bxu���  �          @�@>{���H��  �e=qC��f@>{?�{��(��]Q�A�                                    Bxu��L  �          @�ff@*=q��=q��G��iC���@*=q?������o�\A�ff                                    Bxu���  �          @��?��\�9����{�e=qC�ٚ?��\>\)��\)Ǯ@���                                    Bxu�Ř  �          @��\?h���#33���\�w�RC���?h��?�\����A�=q                                    Bxu��>  �          @��?�=q�!���ff�f�
C��
?�=q>���G�G�Ab=q                                    Bxu���  �          @��þ���o\)��p��M(�C�޸����5��{¦\C�"�                                    Bxu��  �          @�z�>���g
=���R�V�RC���>�׾�ff���
§��C��                                    Bxu� 0  �          @ȣ�?����=q��G��C�Y�?��@�
���\��Bz�R                                    Bxu��  �          @�?�녾�(������C�p�?��@333��=q�n(�B~
=                                    Bxu�|  �          @�(�?��>�
=���H  AUG�?��@\(�����J�Bz
=                                    Bxu�,"  �          @\?�{>L�����\�f@�?�{@P  ���H�T�RB~��                                    Bxu�:�  �          @�G�?�=q>k���=qL�A
=?�=q@P����=q�T=qB�L�                                    Bxu�In  �          @�?�Q�>k���\)u�A�?�Q�@N{��  �U��B�                                    Bxu�X  �          @���?���?\(���G�33B\)?���@i����  �?��B���                                    Bxu�f�  �          @���?W
=��G���{£(�C�8R?W
=@:=q��z��i{B��                                    Bxu�u`  �          @��R>��
������p�«33C��
>��
@/\)��
=�s�B���                                    Bxu��  �          @��>��R��{���\C��H>��R?�����B�Ǯ                                    Bxu���  �          @�  >B�\�6ff���
�h��C��f>B�\<#�
��°  @e�                                    Bxu��R  
�          @�\)>����J�H���X��C���>��;�Q���¨�3C�8R                                    Bxu���  �          @��R?�33�4z���  �_��C�)?�33���
����C�ff                                    Bxu���  �          @�ff?���6ff��p��Z33C��q?����Q���Q�Q�C�)                                    Bxu��D  �          @�=q?\)�%���\�o�C�{?\)>aG���Q�§A��                                    Bxu���  �          @���@%��녿У����C�� @%�333�j=q�/33C�^�                                    Bxu��  �          @�{@%�P���33���C�<)@%�Ǯ�j�H�J{C�y�                                    Bxu��6  �          @��@.{����@,��A�(�C�޸@.{�������=qC��                                    Bxu��  �          @�Q�@K����?�=qA/\)C�P�@K������G��t��C���                                    Bxu��  �          @�G�@S33��=q��Q�h��C��@S33�u��R��
=C�Y�                                    Bxu�%(  �          @��
@y����ff?�Q�A;�C�l�@y����ff�����=p�C�o\                                    Bxu�3�  �          @���@e�����?���A'�C�)@e������R�i�C�z�                                    Bxu�Bt  �          @�ff@X����{?�ffAK\)C��\@X����p���{�T  C���                                    Bxu�Q  �          @�p�@p  ���?aG�A	��C���@p  ���ÿ�{�|��C��R                                    Bxu�_�  T          @��@qG���(�=L��>�ffC�W
@qG��p  �G����HC���                                    Bxu�nf  �          @�ff@`  ��R�i���!�RC�S3@`  �B�\����N=qC�w
                                    Bxu�}  �          @�  @aG��X���B�\���C��@aG����
���
�EC�                                      Bxu���  �          @���@U��~�R�"�\���C���@U���\��Q��=��C�<)                                    Bxu��X  �          @��@N{��(��'��ԸRC�H@N{�Q������BC�E                                    Bxu���  �          @�(�@l(���Q�����{C�l�@l(��0���h���=qC��{                                    Bxu���  �          @���@b�\��{�����w�C�G�@b�\�<���j�H�(�C�
                                    Bxu��J  �          @��
@N�R���\)�ə�C���@N�R�  ��=q�>ffC��{                                    Bxu���  �          @��@J=q��p��  ���
C�@J=q�%�����8ffC�]q                                    Bxu��  �          @�(�@8Q���p��!��̏\C��@8Q��(���
=�Gz�C��q                                    Bxu��<  �          @���@;�����������C�|)@;��(Q���33�?�\C�{                                    Bxu� �  �          @�p�@A�����p���C��@A��.{�����9�C�
=                                    Bxu��  �          @�{@g����\�n{�  C�*=@g��Z�H�L(���RC�Y�                                    Bxu�.  !          @�ff@������H>W
=@�\C���@����s33�ff��p�C�ff                                    Bxu�,�  T          @�{@w
=����?n{AQ�C���@w
=����\�mC�'�                                    Bxu�;z  �          @�z�@�z���z�?(�@�{C���@�z��u��33��z�C���                                    Bxu�J   
Z          @��@����\��>�{@Z=qC�c�@����G��\�v�HC��f                                    Bxu�X�  
�          @���@�(��Z�H�����C���@�(��(���\)���RC�+�                                    Bxu�gl  �          @���@���i��?z�@��C�
=@���Z=q���f{C��)                                    Bxu�v  
�          @��@����p  >�@�=qC�xR@����[���=q��C��
                                    Bxu���  �          @�
=@�z��a�?\(�A33C���@�z��]p���{�4Q�C��                                    Bxu��^  "          @��
@z�H�qG�?�z�Ah  C�@z�H�|�ͿG���ffC�e                                    Bxu��  "          @��@�Q��}p��+��׮C���@�Q��AG��(���ݮC���                                    Bxu���  T          @���@�=q�x�ÿB�\��p�C�%@�=q�:�H�+���p�C�.                                    Bxu��P  �          @�Q�@n�R��G�?c�
AC��@n�R���
���R�p  C�R                                    Bxu���  T          @�(�@aG��w
=��  �V�HC�*=@aG��(���E��(�C���                                    Bxu�ܜ  T          @��
@����H� ����p�C���@��7
=����>��C�&f                                    Bxu��B  T          @�33@z���
=��������C�3@z��Dz������9{C��                                    Bxu���  "          @��@����R�0����\C���@��	������Z=qC�7
                                    Bxu��  T          @�33@\)�z=q�G���C�7
@\)���H��ff�eQ�C�                                    Bxu�4  �          @��@*�H�W
=�e��C�B�@*�H�}p���=q�m�\C��)                                    Bxu�%�  �          @���@G��QG��s33�.C�]q@G��L����ff(�C�:�                                    